clc
clear

gt_dir = './clean/';
gt_files = dir([gt_dir, '*.png']);

restored_dir = './results/';
restored_files = dir([restored_dir, '*.png']);

len = length(restored_files);

psnr_res = 0;
ssim_res = 0;

psnr_list = zeros(1,len);
ssim_list = zeros(1,len);


for i = 1:len
    
    disp(['evaluating ',num2str(i),'/',num2str(len),' images'])
    
    gt_name = [gt_dir, gt_files(i).name];
    gt = imread(gt_name);    
    
    restored_name = [restored_dir, restored_files(i).name];
    restored = imread(restored_name);

    if size(restored,3)>1         
       tmp = rgb2ycbcr(gt);  
       gt = tmp(:,:,1);     
       
       tmp = rgb2ycbcr(restored);
       restored = tmp(:,:,1);
    end
          
    psnr_value = compute_psnr(restored,gt);   
    ssim_value = compute_ssim(restored,gt,0,0);  
         
    psnr_list(1,i) = psnr_value;
    ssim_list(1,i) = ssim_value;

end

avg_PSNR = sum(psnr_list)/len;
avg_SSIM = sum(ssim_list)/len;
disp(['average PSNR = ',num2str(avg_PSNR),', average SSIM = ',num2str(avg_SSIM)])