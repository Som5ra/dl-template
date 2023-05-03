import torch
import os
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp.grad_scaler import GradScaler
from config_util import get_config

def parse_args():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('config')
    args = parser.parse_args()
    return args

def replace_config_paths(config_file, config):
    subsitutes = dict(
        config_name = os.path.basename(config_file).rsplit('.')[0]
    )
    config.weight_saving_dir = config.weight_saving_dir.format(**subsitutes)


class LossSampler:
    def __init__(self) -> None:
        self._sum_cnt = {}
        self._len_cnt = {}
    
    def update(self, name, val):
        if name not in self._sum_cnt:
            self._sum_cnt[name] = 0
            self._len_cnt[name] = 0

        self._sum_cnt[name] += val
        self._len_cnt[name] += 1
    
    def format_loss(self, ):
        formatted = []
        for name in self._sum_cnt.keys():
            avg_l = self._sum_cnt[name] / self._len_cnt[name]
            avg_loss_str = f"{name} = {avg_l:.5f}"
            formatted.append(avg_loss_str)
        return ', '.join(formatted)
    
    def get_loss(self, ):
        losses = []
        for name in self._sum_cnt.keys():
            avg_l = self._sum_cnt[name] / self._len_cnt[name]
            losses.append((name, avg_l))
        return losses

def main(args):
    # setup logging
    import logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # get config
    config = get_config(args.config)
    replace_config_paths(args.config, config)
    logging.info(f"Loaded config {args.config}" )
    
    # prepare directory
    logging.info(f"Weight saving directory: {config.weight_saving_dir}")
    os.makedirs(config.weight_saving_dir, exist_ok=True)

    # preparing data
    training_ds_loader = DataLoader(
        config.training_set, 
        batch_size=config.batch_size,
        num_workers=config.ds_num_workers,
        prefetch_factor=config.ds_prefetch_factor,
        pin_memory = True,
        shuffle=True
    )
    logging.info("Total training data: {}".format(len(config.training_set)))
    
    # preparing model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config.model.train()
    config.model.to(device)    

    # training
    def save_checkpoint(model, save_path, epoch):
        torch.save({
            'model_state_dict': model.state_dict(),
            'epoch': epoch
        }, save_path)
    
    def load_checkpoint(model, model_path):
        return torch.load(model_path)

    start_epoch = 0
    if config.pretrained_model_path:
        logging.info(f"Loading from pre-trained model: {config.pretrained_model_path}")
        cp = load_checkpoint(config.model, config.pretrained_model_path)

        # allow size mismatch
        state_dict = cp['model_state_dict']
        model_state_dict = config.model.state_dict()
        for k in list(cp['model_state_dict']):
            if k in list(config.model.state_dict()):
                if state_dict[k].shape != model_state_dict[k].shape:
                    logging.warning(f"Dropping parameter from checkpoint: {k}, " f"required shape: {model_state_dict[k].shape}, " f"loaded shape: {state_dict[k].shape}")
                    del state_dict[k]


        config.model.load_state_dict(state_dict, strict=False)
        # start_epoch = cp['epoch']
        
    summary_writer = SummaryWriter(log_dir=os.path.join(config.weight_saving_dir, "tf_logs"))
    is_fp16 = config.fp16
    scaler = GradScaler(enabled=is_fp16)

    for epoch in range(start_epoch, start_epoch + config.max_epoch):
        running_loss = LossSampler()
        epoch_step = 0
        for data in (pbar := tqdm(training_ds_loader, total = len(training_ds_loader))): 
            epoch_step += 1

            pbar.set_description(f"Epoch {epoch + 1}/{start_epoch + config.max_epoch}")
            

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)


            # forward + backward + optimize
            with autocast(enabled=is_fp16):
                outputs = config.model(inputs)

                if isinstance(config.loss, dict):
                    loss = 0
                    for k, l in config.loss.items():
                        l_vals = l(outputs, labels) 
                        running_loss.update('k', l_vals)
                        loss += l_vals * config.loss_weights.get(k, 1.0)

                    running_loss.update('loss', loss)


                else:
                    loss = config.loss(outputs, labels)
                    running_loss.update('loss', loss)

                loss /= config.gradient_accum

            scaler.scale(loss).backward()

            if epoch_step % config.gradient_accum == 0:
                scaler.step(optimizer=config.optimizer)
                scaler.update()
                config.optimizer.zero_grad()
            
            
            # if is_fp16:
            #     scaler.scale(loss).backward()
            #     scaler.step(optimizer=config.optimizer)
            #     scaler.update()
            # else:
            #     loss.backward()
            #     config.optimizer.step()


            pbar.set_postfix_str(running_loss.format_loss())
        

        # tensorboard
        summary_writer.add_scalar('learning_rate', config.optimizer.param_groups[0]["lr"], global_step=epoch)
        for loss_name, val in running_loss.get_loss():
            summary_writer.add_scalar(loss_name, val, global_step=epoch)
        summary_writer.flush()
        

        if epoch % config.save_freq == 0:
            save_checkpoint(config.model,save_path=f'{config.weight_saving_dir}/epoch_{epoch}.pth', epoch=epoch)
        save_checkpoint(config.model,save_path=f'{config.weight_saving_dir}/epoch_latest.pth', epoch=epoch)


        logging.info(f"Epoch {epoch} - {running_loss.format_loss()}; lr={config.optimizer.param_groups[0]['lr']}")

        # finishing
        config.scheduler.step()

        
    



if __name__ == '__main__':

    main(parse_args())