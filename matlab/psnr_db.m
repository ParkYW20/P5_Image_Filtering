function p = psnr_db(ref, img)
% (제공) PSNR [dB], 8비트 기준 (결과는 0~255로 잘라서 비교)
e = double(ref(:)) - min(max(double(img(:)), 0), 255);
p = 10*log10(255^2/mean(e.^2));
end
