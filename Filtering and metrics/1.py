import sys
import numpy as np
from skimage import io

def load_image(file_path):
    """
    Загрузка изображения в градациях серого.
    Возвращает изображение с типом данных float64 в диапазоне [0, 255].
    """
    image = io.imread(file_path, as_gray=True).astype(np.float64)
    return image

def save_image(image, file_path):
    """
    Сохранение изображения.
    Масштабирование значений до [0, 255], обрезка и преобразование в uint8.
    """
    image = np.clip(image, 0, 255).astype(np.uint8)
    io.imsave(file_path, image)

def median_filter(image, rad):
    """
    Применение медианного фильтра к изображению.
    
    :param image: Входное изображение в диапазоне [0, 255]
    :param rad: Радиус фильтра
    :return: Отфильтрованное изображение
    """
    padded_image = np.pad(image, rad, mode='edge')
    filtered_image = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            window = padded_image[i:i + 2 * rad + 1, j:j + 2 * rad + 1]
            filtered_image[i, j] = np.median(window)
    return filtered_image

def gaussian_kernel(size, sigma):
    """
    Создание ядра Гаусса.
    
    :param size: Размер ядра
    :param sigma: Стандартное отклонение
    :return: Нормализованное гауссово ядро
    """
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / np.sum(kernel)

def gaussian_filter_custom(image, sigma_d):
    """
    Применение гауссова фильтра к изображению.
    
    :param image: Входное изображение в диапазоне [0, 255]
    :param sigma_d: Стандартное отклонение
    :return: Отфильтрованное изображение
    """
    radius = int(np.ceil(3 * sigma_d))
    size = 2 * radius + 1
    kernel = gaussian_kernel(size, sigma_d)
    padded_image = np.pad(image, radius, mode='edge')
    filtered_image = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            window = padded_image[i:i + size, j:j + size]
            filtered_image[i, j] = np.sum(window * kernel)
    return filtered_image

def mse(image1, image2):
    """
    Вычисление среднеквадратичной ошибки между двумя изображениями.
    
    :param image1: Первое изображение
    :param image2: Второе изображение
    :return: Значение MSE
    """
    return np.mean((image1 - image2) ** 2)

def psnr(image1, image2):
    """
    Вычисление пиксового отношения сигнала к шуму (PSNR).
    
    :param image1: Первое изображение
    :param image2: Второе изображение
    :return: Значение PSNR
    """
    mse_value = mse(image1, image2)
    if mse_value == 0:
        return float('inf')
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse_value))

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
    mu_x = np.mean(image1)
    mu_y = np.mean(image2)
    sigma_x = np.var(image1)
    sigma_y = np.var(image2)
    sigma_xy = np.mean((image1 - mu_x) * (image2 - mu_y))
    
    c1 = (k1 * L) ** 2
    c2 = (k2 * L) ** 2
    
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
    
    return numerator / denominator

def bilateral_filter(image, size=5, sigma_d=10.0, sigma_r=25.0):
    """
    Применение билинейного фильтра к изображению.
    
    :param image: Входное изображение в диапазоне [0, 255]
    :param size: Размер ядра
    :param sigma_d: Стандартное отклонение для пространственных весов
    :param sigma_r: Стандартное отклонение для интенсивностных весов
    :return: Отфильтрованное изображение
    """
    kernel_size = int(size)
    pad_width = kernel_size // 2
    padded_image = np.pad(image, pad_width, mode='edge').astype(np.float32)
    h, w = image.shape

    # Предварительное вычисление пространственных весов
    lim = (size - 1) / (2 * sigma_d)
    ax = np.linspace(-lim, lim, kernel_size)
    corex = np.exp(-0.5 * (ax ** 2))
    spatial_weights = np.outer(corex, corex)

    filter_image = np.zeros_like(image, dtype=np.float32)

    for i in range(h):
        for j in range(w):
            window = padded_image[i:i + kernel_size, j:j + kernel_size]
            center_pixel = padded_image[i + pad_width, j + pad_width]
            intensity_weights = np.exp(-0.5 * ((window - center_pixel) / sigma_r) ** 2)
            weights = spatial_weights * intensity_weights
            bilat = np.sum(weights * window) / np.sum(weights)
            filter_image[i, j] = bilat

    filter_image = np.clip(filter_image, 0, 255).astype(np.uint8)
    return filter_image

def main():
    command = sys.argv[1]

    if command in ['median', 'gauss', 'bilateral']:
        if command == 'median':
            try:
                rad = int(sys.argv[2])
                if rad <= 0:
                    raise ValueError
            except ValueError:
                print("rad должно быть целым положительным числом.")
                return
            input_file = sys.argv[3]
            output_file = sys.argv[4]
            image = load_image(input_file)
            result_image = median_filter(image, rad)
            save_image(result_image, output_file)
            
        elif command == 'gauss':
            try:
                sigma_d = float(sys.argv[2])
                if sigma_d <= 0:
                    raise ValueError
            except ValueError:
                print("sigma_d должно быть вещественным положительным числом.")
                return
            input_file = sys.argv[3]
            output_file = sys.argv[4]
            image = load_image(input_file)
            result_image = gaussian_filter_custom(image, sigma_d)
            save_image(result_image, output_file)
            print(f"Гауссов фильтр применен успешно. Результат сохранен в {output_file}.")
    
        elif command == 'bilateral':
            try:
                sigma_d = float(sys.argv[2])
                sigma_r = float(sys.argv[3])
                if sigma_d <= 0 or sigma_r <= 0:
                    raise ValueError
            except ValueError:
                print("sigma_d и sigma_r должны быть положительными числами.")
                return
            input_file = sys.argv[4]
            output_file = sys.argv[5]
            image = load_image(input_file)
            size = int(2 * np.ceil(3 * sigma_d) + 1)
            result_image = bilateral_filter(image, size=size, sigma_d=sigma_d, sigma_r=sigma_r)
            save_image(result_image, output_file)
            print(f"Bilateral фильтр применен успешно. Результат сохранен в {output_file}.")

    elif command in ['mse', 'psnr', 'ssim']:
        try:
            input_file_1 = sys.argv[2]
            input_file_2 = sys.argv[3]
            image1 = load_image(input_file_1)
            image2 = load_image(input_file_2)
        except IndexError:
            print("Недостаточно аргументов для вычисления метрик.")
            return

        if image1.shape != image2.shape:
            print("Изображения должны иметь одинаковые размеры.")
            return

        if command == 'mse':
            mse_value = mse(image1, image2)
            print(f"{mse_value}")
        elif command == 'psnr':
            psnr_value = psnr(image1, image2)
            print(f"{psnr_value}")
        elif command == 'ssim':
            ssim_value = ssim(image1, image2)
            print(f"{ssim_value}")
    else:
        print("Неизвестная команда.")
        return

if __name__ == "__main__":
    main()