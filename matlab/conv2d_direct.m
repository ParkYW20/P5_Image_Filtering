function y = conv2d_direct(img, h)
% 2D 컨볼루션 직접 구현 (conv2/imfilter/filter2 사용 금지)
%   y(m,n) = sum_i sum_j h(i,j) x(m-i, n-j),  출력 크기 = 입력 크기, 경계 = 가장자리 화소 복제, 커널 크기 홀수
%   커널을 뒤집어서(rot90(h,2)) 곱한다 (컨볼루션 ≠ 상관)
% 힌트(빠른 방법): 패딩한 영상에서 커널 원소 개수만큼 '밀린 영상'을 잘라 가중합
%   xp = img([ones(1,ry) 1:H H*ones(1,ry)], [ones(1,rx) 1:W W*ones(1,rx)]);   % 가장자리 복제 패딩
%   for i, j: y = y + hf(i,j) * xp(i:i+H-1, j:j+W-1);
% TODO
error('TODO:notImplemented', 'conv2d_direct 미구현');
end
