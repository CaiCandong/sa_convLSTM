import os
import matplotlib.pyplot as plt

def save_images(targets,outputs,cache_dir,epoch,batch_idx):
    """"
        绘图代码
    """
    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)
    _, axarr = plt.subplots(2, targets.shape[1], figsize=(targets.shape[1] * 5, 10))
    for t in range(targets.shape[1]):
        axarr[0][t].imshow(targets[0, t, 0].detach().cpu().numpy(), cmap='gray')
        axarr[1][t].imshow(outputs[0, t, 0].detach().cpu().numpy(), cmap='gray')
    plt.savefig(os.path.join(cache_dir, '{:03d}_{:05d}.png'.format(epoch, batch_idx)))
    plt.close()   