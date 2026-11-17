function s = sharpness(img)
% (제공) 선명도 = 평균 기울기 크기 (중앙 차분, Python np.gradient 와 동일)
f = min(max(double(img), 0), 255);
[gx, gy] = gradient(f); s = mean(hypot(gx(:), gy(:)));
end
