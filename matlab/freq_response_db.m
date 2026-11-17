function HdB = freq_response_db(h, n)
% (제공) 2D 커널의 주파수 응답 크기 [dB], 중앙이 DC (n x n 영점 채움 DFT)
if nargin < 2, n = 256; end
HdB = 20*log10(abs(fftshift(fft2(h, n, n))) + 1e-12);
end
