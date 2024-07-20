function img = preprocessImage(filepath, targetSize)
    img = imread(filepath);
    if size(img, 3) == 3
        img = rgb2gray(img);
    end
    img = imresize(img, targetSize(1:2));
    img = im2single(img); % Conversie la tipul de date necesar
end

