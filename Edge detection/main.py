import numpy as np
import sys
from skimage import io

def load_image(file_path):
    """
    Загрузка изображения. Если изображение цветное, используется только первый канал.
    Возвращает изображение с типом данных float32 в диапазоне [0, 255].
    """
    image = io.imread(file_path)
    if len(image.shape) == 3:
        image = image[:, :, 0]
    return image.astype(np.float32)

def save_image(image, file_path):
    """
    Сохранение изображения.
    Масштабирование значений до [0, 255], обрезка и преобразование в uint8.
    """
    image = np.clip(image, 0, 255).round().astype(np.uint8)
    io.imsave(file_path, image)

def mse(image1, image2):
    """
    Вычисление среднеквадратичной ошибки между двумя изображениями.
    
    :param image1: Первое изображение
    :param image2: Второе изображение
    :return: Значение MSE
    """
    image_1 = np.copy(image1)
    image_2 = np.copy(image2)
    return np.mean((image_1 - image_2) ** 2)

def psnr(image1, image2):
    """
    Вычисление пиксового отношения сигнала к шуму (PSNR).
    
    :param image1: Первое изображение
    :param image2: Второе изображение
    :return: Значение PSNR
    """
    image_1 = np.copy(image1)
    image_2 = np.copy(image2)
    eps = 1e-11
    mse_value = max(mse(image_1, image_2), eps)
    if mse_value == 0:
        return float('inf')
    max_pixel = 255.0
    return 10 * np.log10(max_pixel**2 / mse_value)

def ssim(image1, image2, k1=0.01, k2=0.03, L=255.0):
    """
    Вычисление структурного сходства между двумя изображениями (SSIM).
    
    :param image1: Первое изображение
    :param image2: Второе изображение
    :param k1: Константа для стабильности
    :param k2: Константа для стабильности
    :param L: Динамический диапазон яркости (255 для uint8)
    :return: Значение SSIM
    """
    image_1 = np.copy(image1)
    image_2 = np.copy(image2)
    mu_x = np.mean(image_1)
    mu_y = np.mean(image_2)
    sigma_x = np.mean((image_1 - mu_x)**2)
    sigma_y = np.mean((image_2 - mu_y)**2)
    sigma_xy = np.mean((image_1 - mu_x) * (image_2 - mu_y))
    
    c1 = (k1 * L) ** 2
    c2 = (k2 * L) ** 2
    
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
    
    return numerator / denominator

def gaussian_derivative_kernel(size, sigma, axis):
    """
    Создание ядра производной функции Гаусса.
    
    :param size: Размер ядра
    :param sigma: Стандартное отклонение
    :param axis: Ось производной (0 для x, 1 для y)
    :return: Ядро производной функции Гаусса
    """
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    if axis == 0:
        kernel = -xx * np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    else:
        kernel = -yy * np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel

def convolve(image, kernel):
    """
    Свёртка изображения с ядром.
    """
    kernel = np.flipud(np.fliplr(kernel))
    output = np.zeros_like(image)
    padded_image = np.pad(image, ((kernel.shape[0] // 2, kernel.shape[0] // 2), (kernel.shape[1] // 2, kernel.shape[1] // 2)), mode='edge')
    for x in range(image.shape[1]):
        for y in range(image.shape[0]):
            output[y, x] = (kernel * padded_image[y: y + kernel.shape[0], x: x + kernel.shape[1]]).sum()
    return output

def gradient_magnitude(image, sigma):
    size = int(2 * (np.ceil(3 * sigma)) + 1)
    gx = gaussian_derivative_kernel(size, sigma, axis=0)
    gy = gaussian_derivative_kernel(size, sigma, axis=1)
    ix = convolve(image, gx)
    iy = convolve(image, gy)
    magnitude = np.sqrt(ix**2 + iy**2)
    gmax = magnitude.max()
    magnitude = (magnitude / gmax) * 255
    return magnitude, ix, iy

def non_max_suppression(magnitude, ix, iy):
    """Подавление немаксимумов."""
    rows, cols = magnitude.shape
    suppressed = np.zeros_like(magnitude, dtype=np.float32)
    angle = np.rad2deg(np.arctan2(iy, ix)) + 180  # Угол в градусах [0, 360]

    padded_magnitude = np.pad(magnitude, ((1, 1), (1, 1)), mode='edge')
    padded_angle = np.pad(angle, ((1, 1), (1, 1)), mode='edge')

    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            # Определение направления градиента (округление до 4 направлений)
            direction = np.round(padded_angle[i, j] / 45) % 4  # Дискретизация углов

            if direction == 0:  # Восток-запад (горизонтальный)
                neighbors = padded_magnitude[i, j-1], padded_magnitude[i, j+1]
            elif direction == 1:  # Северо-восток - Юго-запад
                neighbors = padded_magnitude[i-1, j-1], padded_magnitude[i+1, j+1]
            elif direction == 2:  # Север-юг (вертикальный)
                neighbors = padded_magnitude[i-1, j], padded_magnitude[i+1, j]
            elif direction == 3:  # Северо-запад - Юго-восток
                neighbors = padded_magnitude[i-1, j+1], padded_magnitude[i+1, j-1]

            if padded_magnitude[i, j] >= max(neighbors):
                suppressed[i-1, j-1] = padded_magnitude[i, j]

    return suppressed

def hysteresis(suppressed, low_threshold, high_threshold):
    strong = 255
    weak = 75
    result = np.zeros_like(suppressed)
    strong_i, strong_j = np.where(suppressed >= high_threshold)
    zeros_i, zeros_j = np.where(suppressed < low_threshold)
    weak_i, weak_j = np.where((suppressed <= high_threshold) & (suppressed >= low_threshold))
    result[strong_i, strong_j] = strong
    result[weak_i, weak_j] = weak

    def is_strong_neighbor(i, j):
        return ((result[i + 1, j - 1] == strong) or (result[i + 1, j] == strong) or (result[i + 1, j + 1] == strong)
                or (result[i, j - 1] == strong) or (result[i, j + 1] == strong)
                or (result[i - 1, j - 1] == strong) or (result[i - 1, j] == strong) or (result[i - 1, j + 1] == strong))

    changed = True
    while changed:
        changed = False
        for i in range(1, suppressed.shape[0] - 1):
            for j in range(1, suppressed.shape[1] - 1):
                if result[i, j] == weak:
                    if is_strong_neighbor(i, j):
                        result[i, j] = strong
                        changed = True

    result[result != strong] = 0  # Установить все не-сильные пиксели в 0
    return result

def canny_edge_detection(image, sigma, high_threshold_ratio, low_threshold_ratio):
    magnitude, ix, iy = gradient_magnitude(image, sigma)
    suppressed = non_max_suppression(magnitude, ix, iy)
    high_threshold = 255 * high_threshold_ratio
    low_threshold = 255 * low_threshold_ratio
    result = hysteresis(suppressed, low_threshold, high_threshold)
    return result


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py (command) [parameters...]")
        return

    command = sys.argv[1]

    if command in ["grad", "nonmax", "canny"]:
        if len(sys.argv) < 5:
            print("Usage: python main.py (command) [parameters...] (input_image) (output_image)")
            return

        input_image_path = sys.argv[-2]
        output_image_path = sys.argv[-1]
        image = load_image(input_image_path)

        if command == "grad":
            sigma = float(sys.argv[2])
            magnitude, _, _ = gradient_magnitude(image, sigma)
            save_image(magnitude, output_image_path)
        elif command == "nonmax":
            sigma = float(sys.argv[2])
            magnitude, ix, iy = gradient_magnitude(image, sigma)
            suppressed = non_max_suppression(magnitude, ix, iy)
            save_image(suppressed, output_image_path)
        elif command == "canny":
            sigma = float(sys.argv[2])
            high_threshold_ratio = float(sys.argv[3])
            low_threshold_ratio = float(sys.argv[4])
            result = canny_edge_detection(image, sigma, high_threshold_ratio, low_threshold_ratio)
            save_image(result, output_image_path)
    elif command in ["mse", "psnr", "ssim"]:
        if len(sys.argv) < 4:
            print("Usage: python main.py (command) (input_image_1) (input_image_2)")
            return

        input_file_1 = sys.argv[2]
        input_file_2 = sys.argv[3]
        image1 = load_image(input_file_1)
        image2 = load_image(input_file_2)

        if command == "mse":
            mse_value = mse(image1, image2)
            print(f"MSE: {mse_value}")
        elif command == "psnr":
            psnr_value = psnr(image1, image2)
            print(f"PSNR: {psnr_value}")
        elif command == "ssim":
            ssim_value = ssim(image1, image2, L=255.0)
            print(f"SSIM: {ssim_value}")
    else:
        print("Unknown command")

if __name__ == "__main__":
    main()