% P5 영상 필터링 — 공통 평가 스크립트 (MATLAB/Octave, 수정 불필요)
here = fileparts(mfilename('fullpath')); data = fullfile(here, '..', 'data');
test_csv = fullfile(data, 'test_list.csv');        % 다른 시험 세트 평가 시 이 두 줄만 변경
test_dir = fullfile(data, 'images_test');

do_plot = true;
if exist('OCTAVE_VERSION', 'builtin'), do_plot = ~isempty(available_graphics_toolkits()); end
outdir = fullfile(here, '..', 'results'); if ~exist(outdir, 'dir'), mkdir(outdir); end
isTodo = @(e) strcmp(e.identifier, 'TODO:notImplemented');

fprintf('\n--- 1. 구현 검증 ---\n');
rand('seed', 0); x = rand(64, 80)*255; h = rand(3, 5)*2 - 1; %#ok<RAND>
try
    y = conv2d_direct(x, h);
    xp = x([1 1:64 64], [1 1 1:80 80 80]); ref = conv2(xp, h, 'valid');   % 가장자리 복제 + 컨볼루션
    fprintf('conv2d_direct : 크기 %dx%d (기대 64x80), 기준 대비 최대 오차 = %.2e\n', size(y, 1), size(y, 2), max(abs(y(:) - ref(:))));
    refc = conv2(xp, rot90(h, 2), 'valid');
    if max(abs(y(:) - ref(:))) > 1e-6 && max(abs(y(:) - refc(:))) < 1e-6
        fprintf('                → 상관(correlation)과 일치합니다. 커널을 뒤집었는지 확인하세요.\n');
    end
catch err
    if ~isTodo(err), rethrow(err); end
    fprintf('conv2d_direct : 미구현 → 이후 평가 불가\n'); return;
end
try
    b = box_kernel(5); g = gaussian_kernel(1.2);
    fprintf('box_kernel(5)       : 크기 %dx%d, 합 %.6f\n', size(b, 1), size(b, 2), sum(b(:)));
    fprintf('gaussian_kernel(1.2): 크기 %dx%d (기대 9x9), 합 %.6f, 대칭 오차 %.1e, 중앙값 %.4f (연속 근사 %.4f)\n', ...
        size(g, 1), size(g, 2), sum(g(:)), max(max(abs(g - g'))), g(5, 5), 1/(2*pi*1.2^2));
catch err
    if ~isTodo(err), rethrow(err); end
    fprintf('커널 : 미구현 → 이후 평가 불가\n'); return;
end
img = double(imread(fullfile(data, 'images_clean', 'camera.png')));
g2 = gaussian_kernel(2.0); k1 = sum(g2, 2)';
tic; yd = conv2d_direct(img, g2); td = toc;
fprintf('속도 (512x512, 가우시안 σ=2 → %dx%d): 2D 직접 %.3f s', size(g2, 1), size(g2, 2), td);
try
    tic; ys = conv2d_separable(img, k1); ts = toc;
    fprintf(', 분리형 %.3f s (결과 차이 %.1e), 곱셈 수/화소 %d → %d\n', ts, max(abs(ys(:) - yd(:))), numel(g2), 2*numel(k1));
catch err
    if ~isTodo(err), rethrow(err); end
    fprintf(', 분리형: (선택 과제) 미구현\n');
end

fid = fopen(test_csv); fgetl(fid); C = textscan(fid, '%s %s %s %f', 'Delimiter', ','); fclose(fid);
files = C{1}; cleans = C{2}; kinds = C{3};
names = {'box3', 'box5', 'box7', 'gau0.8', 'gau1.2', 'gau2.0', 'med3', 'med5'};
fprintf('\n--- 2. 블러링으로 잡음 제거: PSNR [dB] (원본 대비) ---\n%-28s %7s', 'test image', '입력');
fprintf(' %7s', names{:}); fprintf('\n');
for t = 1:numel(files)
    if ~any(strcmp(kinds{t}, {'gauss', 'sp'})), continue; end
    c = double(imread(fullfile(data, 'images_clean', cleans{t}))); f = double(imread(fullfile(test_dir, files{t})));
    out = {conv2d_direct(f, box_kernel(3)), conv2d_direct(f, box_kernel(5)), conv2d_direct(f, box_kernel(7)), ...
           conv2d_direct(f, gaussian_kernel(0.8)), conv2d_direct(f, gaussian_kernel(1.2)), conv2d_direct(f, gaussian_kernel(2.0)), [], []};
    for m = 1:2
        try, out{6+m} = median_filter_direct(f, 2*m+1); catch err, if ~isTodo(err), rethrow(err); end, end
    end
    fprintf('%-28s %7.2f', files{t}, psnr_db(c, f));
    for k = 1:numel(out)
        if isempty(out{k}), fprintf(' %7s', '-'); else, fprintf(' %7.2f', psnr_db(c, out{k})); end
    end
    fprintf('\n');
end
fprintf('(med3/med5 = 선택 과제 메디안 필터)\n');

fprintf('\n--- 3. 샤프닝으로 블러 복원: 언샤프 마스킹 (mask σ = 2.0) ---\n');
alphas = [0.5 1 2 4];
fprintf('%-28s %8s %8s', 'test image', '', '입력'); fprintf('   α=%-4.1f', alphas); fprintf('\n');
for t = 1:numel(files)
    if ~strcmp(kinds{t}, 'blur'), continue; end
    c = double(imread(fullfile(data, 'images_clean', cleans{t}))); f = double(imread(fullfile(test_dir, files{t})));
    try
        o = arrayfun(@(a) unsharp_mask(f, 2.0, a), alphas, 'UniformOutput', false);
    catch err
        if ~isTodo(err), rethrow(err); end
        fprintf('unsharp_mask: 미구현\n'); break;
    end
    s0 = sharpness(c);
    fprintf('%-28s %8s %8.2f', files{t}, 'PSNR', psnr_db(c, f)); fprintf(' %8.2f', cellfun(@(v) psnr_db(c, v), o)); fprintf('\n');
    fprintf('%-28s %8s %8.2f', '', '선명도비', sharpness(f)/s0); fprintf(' %8.2f', cellfun(@(v) sharpness(v)/s0, o)); fprintf('\n');
end

if do_plot
    fig = figure('Visible', 'off', 'Position', [100 100 1200 350]); wn = (-128:127)/256;
    Hb = freq_response_db(box_kernel(5)); Hg = freq_response_db(gaussian_kernel(1.2));
    subplot(1, 3, 1); plot(wn, Hb(129, :), wn, Hg(129, :)); ylim([-60 12]); grid on;
    legend('box 5x5', 'gaussian \sigma=1.2'); xlabel('cycles/pixel'); ylabel('dB'); title('frequency response (center row)');
    subplot(1, 3, 2); imagesc(wn, wn, Hb, [-60 0]); axis image; title('|H| box 5x5');
    subplot(1, 3, 3); imagesc(wn, wn, Hg, [-60 0]); axis image; title('|H| gaussian \sigma=1.2');
    print(fig, fullfile(outdir, 'freq_responses_matlab.png'), '-dpng'); close(fig);
end
fprintf('\n결과 저장 → %s\n', outdir);
