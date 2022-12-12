from loader import *
from measure import *
from networks import *

if __name__ == "__main__":

    def denormalize_(image, norm_range_max = 50000.0, norm_range_min = 0.0):
        image = image * (norm_range_max - norm_range_min) + norm_range_min
        return image

    def trunc(mat, trunc_min = -1024.0, trunc_max = 3072.0):
        mat[mat <= trunc_min] = trunc_min
        mat[mat >= trunc_max] = trunc_max
        return mat

    total_iters = 46100
    save_iters = 461
    trunc_max = 3072.0
    trunc_min = -1024.0
    pred_psnr_avg, pred_ssim_avg, pred_rmse_avg = 0.0, 0.0, 0.0

    for epoch in range(0, 100):
        save_path = 'save/'
        if total_iters % save_iters == 0 and total_iters != 0:    # 修改
            dataset_ = ct_dataset(mode='test', load_mode=0, saved_path='E:/limei/test(2)/',
                                    test_patient='LDCT', patch_n=None,
                                    patch_size=None, transform=False)
            data_loader = DataLoader(dataset=dataset_, batch_size=1, shuffle=True, num_workers=0)

            WGAN_VGG_generator1 = RED_CNN().cuda()
            f = os.path.join(save_path, 'REDCNN_{}iter.ckpt'.format((epoch + 1) * save_iters))  # 修改
            WGAN_VGG_generator1.load_state_dict(torch.load(f))

            with torch.no_grad():
                for i, (x, y) in enumerate(data_loader):
                    shape_ = x.shape[-1]
                    x = x.unsqueeze(0).float().cuda()
                    y = y.unsqueeze(0).float().cuda()

                    pred = WGAN_VGG_generator1(x)
                    x1 = trunc(denormalize_(x))
                    y1 = trunc(denormalize_(y))
                    pred1 = trunc(denormalize_(pred))
                    data_range = trunc_max - trunc_min
                    original_result, pred_result = compute_measure(x1, y1, pred1, data_range)

                    pred_psnr_avg += pred_result[0]
                    pred_ssim_avg += pred_result[1]
                    pred_rmse_avg += pred_result[2]

            #########################################################
            # 日志文件
            # with open('Loss.txt', 'a') as f:
            #     f.write('ITER:%d loss:%.20f' % (total_iters, loss) + '\n')
            #     f.close()

            with open('./save/pred_psnr_avg_0.txt', 'a') as f:
                f.write('EPOCH:%d loss:%.20f' % (epoch, pred_psnr_avg / len(data_loader)) + '\n')
                f.close()

            with open('./save/pred_ssim_avg_0.txt', 'a') as f:
                f.write('EPOCH:%d loss:%.20f' % (epoch, pred_ssim_avg / len(data_loader)) + '\n')
                f.close()

            with open('./save/pred_rmse_avg_0.txt', 'a') as f:
                f.write('EPOCH:%d loss:%.20f' % (epoch, pred_rmse_avg / len(data_loader)) + '\n')
                f.close()
            pred_psnr_avg = 0
            pred_ssim_avg = 0
            pred_rmse_avg = 0
            #########################################################
        else:
            continue

