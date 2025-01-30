import argparse  # модуль (библиотека) для обработки параметров коммандной строки
import numpy as np  # модуль для работы с массивами и векторных вычислений
import skimage.io  # модуль для обработки изображений, подмодуль для чтения и записи
from matplotlib import pyplot as plt


# в некоторых модулях некоторые подмодули надо импортировать вручную, а не просто "import module" и потом в коде писать "module.submodule.something..."


def fltr(image,kernel,size=1,/,**kwards):
    kernel_size=size
    #print(kwards)
    image1=np.copy(image).astype(np.float32)
    #print(image[:,[-1]])
    image1=np.insert(image1,[0]*(kernel_size//2),image1[:,[0]],axis=1)
    image1=np.insert(image1,[-1]*(kernel_size//2),image1[:,[-1]],axis=1)
    #print(np.concatenate(((image[:,[-1]],image1[:,[-1]])),axis=1))
    image1=np.insert(image1,[0]*(kernel_size//2),image1[[0]],axis=0)
    image1=np.insert(image1,[-1]*(kernel_size//2),image1[[-1]],axis=0)
    #print(image1[:,[-1]])
    h,w=image.shape

    filter_image=np.zeros(img.shape,dtype=np.float32)
    for i in range(h):
        for j in range(w):
            window=image1[i:i+kernel_size,j:j+kernel_size]
            filter_image[i][j]=kernel(window,size,**kwards)
    filter_image=(filter_image.clip(0,255)).astype(np.uint8)
    return filter_image


def gaussian_kernel(img,size,sigma):
    """
    Построение ядра фильтра Гаусса.

    @param  size  int    размер фильтра
    @param  sigma float  параметр размытия
    @return numpy array  фильтр Гаусса размером size x size
    """

    lim=(size-1)/(2*sigma)
   # print(sigma)
    corex=np.linspace(-lim,lim,size)
    corex=np.exp(-0.5*corex*corex)
    gauss=corex[None,:]*corex[:,None]/(2*np.pi*sigma**2)
    gauss/=np.sum(gauss)

    return np.sum(img*gauss)


def median_kernel(img,rad):

    pixels=img.reshape(-1)
    pixels=np.sort(pixels)

    medium,odd=divmod(pixels.shape[0],2)
    median=np.mean(pixels[medium-(not(odd)):medium+1])
    return median

def bilateral_kernel(img,size,sigma_d,sigma_r):
    lim=(size-1)/(2*sigma_d)
   # print(sigma)
    corex=np.linspace(-lim,lim,size)
    corex=np.exp(-0.5*corex*corex)
    corex_i=np.exp(-0.5*((img-img[size//2][size//2])/(sigma_r))**2)
    w=corex[None,:]*corex[:,None]*corex_i
    bilat=(np.sum(w*img))/(np.sum(w))


    return bilat

def psnr(img1, img2):
    """
    PSNR metric

    @param  img1    numpy array   оригинальное изображение
    @param  img2    numpy array   искаженное изображение
    @return float   PSNR(img1, img2)
    """
    im1=np.copy(img1).astype(np.float32)
    im2=np.copy(img2).astype(np.float32)
    im=im1-im2
    eps=1e-10
    mse=max(np.mean(im*im),eps)
    psnr=10* np.log10(255**2/(mse),dtype=np.float32)
    return psnr


def mse(img1,img2):
        im1=img1.copy().astype(np.float32)
        im2=img2.copy().astype(np.float32)
        res=np.mean((im1-im2)**2)
        return res

def ssim(img1,img2):
    im1=img1.copy().astype(np.float32)
    im2=img2.copy().astype(np.float32)
    k1=0.01
    k2=0.03
    mean1=np.mean(im1)
    mean2=np.mean(im2)
    var1=np.mean((im1-mean1)**2)
    var2=np.mean((im2-mean2)**2)
    cov=np.mean((im1-mean1)*(im2-mean2))
    c1=(k1*255)**2
    c2=(k2*255)**2
    res=(2*mean1*mean2+c1)*(2*cov+c2)/(mean1**2+mean2**2+c1)/(var1+var2+c2)
    return res

if __name__ == '__main__':  # если файл выполняется как отдельный скрипт (python script.py), то здесь будет True. Если импортируется как модуль, то False. Без этой строки весь код ниже будет выполняться и при импорте файла в виде модуля (например, если захотим использовать эти функции в другой программе), а это не всегда надо.
    # получить значения параметров командной строки
    parser = argparse.ArgumentParser(  # не все параметры этого класса могут быть нужны; читайте мануалы на docs.python.org, если интересно
        prog='ProgramName',
        description='What the program does',
        epilog='Text at the bottom of help',  # в конце списка параметров и при создании list, tuple, dict и set можно оставлять запятую, чтобы можно было удобно комментить или добавлять новые строчки без добавления и удаления новых запятых
    )
    parser.add_argument('command', help='Command description')  # add_argument() поддерживает параметры вида "-p 0.1", может сохранять их как числа, строки, включать/выключать переменные True/False ("--activate-someting"), поддерживает задание значений по умолчанию; полезные параметры: action, default, dest - изучайте, если интересно
    parser.add_argument('parameters', nargs='*')  # все параметры сохранятся в список: [par1, par2,...] (или в пустой список [], если их нет)
    parser.add_argument('input_file')
    parser.add_argument('output_file')
    args = parser.parse_args()

    # Можете посмотреть, как распознаются разные параметры. Но в самом решении лишнего вывода быть не должно.
    # print('Распознанные параметры:')
    # print('Команда:', args.command)  # между 2 выводами появится пробел
    # print('Её параметры:', args.parameters)
    # print('Входной файл:', args.input_file)
    # print('Выходной файл:', args.output_file)

    img = skimage.io.imread(args.input_file)  # прочитать изображение
    #img = img / 255  # перевести во float и диапазон [0, 1]
    if len(img.shape) == 3:  # оставим только 1 канал (пусть будет 0-й) для удобства: всё равно это ч/б изображение
        img = img[:, :, 0]

    # получить результат обработки для разных комманд
    if args.command == 'gauss':
        sig=float(args.parameters[0])
        size=np.ceil((3*sig)*2).astype(np.int32)
        size+=1-size%2
        res=fltr(img,gaussian_kernel,size,sigma=sig)
        skimage.io.imsave(args.output_file, res)
    elif args.command == 'median':
        rad=int(args.parameters[0])
        size=2*rad+1
        res=fltr(img,median_kernel,size)
        skimage.io.imsave(args.output_file, res)
    elif args.command == 'bilateral':
        sigma_d=float(args.parameters[0])
        sigma_r=float(args.parameters[1])
        size=np.ceil((3*sigma_d)*2).astype(np.int32)
        size+=1-size%2
        res=fltr(img,bilateral_kernel,size,sigma_d=sigma_d,sigma_r=sigma_r)
        skimage.io.imsave(args.output_file, res)

    elif args.command == 'mse':
        img1 = skimage.io.imread(args.output_file)  # прочитать изображение
        if len(img1.shape) == 3:  # оставим только 1 канал (пусть будет 0-й) для удобства: всё равно это ч/б изображение
            img1 = img1[:, :, 0]
        res=mse(img,img1)
        print(res)

    elif args.command == 'psnr':
        img1 = skimage.io.imread(args.output_file)  # прочитать изображение
        if len(img1.shape) == 3:  # оставим только 1 канал (пусть будет 0-й) для удобства: всё равно это ч/б изображение
            img1 = img1[:, :, 0]
        res=psnr(img,img1)
        print(res)
    elif args.command == 'ssim':
        img1 = skimage.io.imread(args.output_file)  # прочитать изображение
        if len(img1.shape) == 3:  # оставим только 1 канал (пусть будет 0-й) для удобства: всё равно это ч/б изображение
            img1 = img1[:, :, 0]
        res=ssim(img,img1)
        print(res)
        #left_x, top_y, width, height = [int(x) for x in args.parameters]  # создать список из сконвертированных параметров и разложить по 4 переменным
        #res = extract(img, left_x, top_y, width, height)

    #elif args.command == 'rotate':
        #direction = args.parameters[0]
        #angle = int(args.parameters[1])
        #res = rotate(img, direction, angle)

    #elif args.command == 'autocontrast':
        #res = autocontrast(img)

    #elif args.command == 'fixinterlace':
        #res = fixinterlace(img)


    # сохранить результат
    #res = np.clip(res, 0, 1)  # обрезать всё, что выходит за диапазон [0, 1]
    #res = np.round(res * 255).astype(np.uint8)  # конвертация в байты







# if __name__ == '__main__':  # если файл выполняется как отдельный скрипт (python script.py), то здесь будет True. Если импортируется как модуль, то False. Без этой строки весь код ниже будет выполняться и при импорте файла в виде модуля (например, если захотим использовать эти функции в другой программе), а это не всегда надо.
#
#
#     img = skimage.io.imread("ex_interlace.bmp")  # прочитать изображение
#
#     img = img / 255  # перевести во float и диапазон [0, 1]
#     if len(img.shape) == 3:  # оставим только 1 канал (пусть будет 0-й) для удобства: всё равно это ч/б изображение
#         img = img[:, :, 0]
#     skimage.io.imshow(img)
#     skimage.io.show()

    # получить результат обработки для разных комманд
    #res = mirror(img, 'cd')
    #res = extract(img, 100, 32, 192, 96)

    #res = rotate(img, 'cw', 720)
    #res = autocontrast(img)
    #res = fixinterlace(img)
    # skimage.io.imshow(res)
    # skimage.io.show()




    # Ещё некоторые полезные штуки в Питоне:

    # l = [1, 2, 3]  # list
    # l = l + [4, 5]  # сцепить списки
    # l = l[1:-2]  # получить кусок списка (slice)

    # Эти тоже можно сцеплять и т.п. - читайте мануалы
    # t = (1, 2, 3)  # tuple, элементы менять нельзя, но можно сцеплять и т.д.
    # s = {1, 'a', None}  # set

    # d = {1: 'a', 2: 'b'}  # dictionary
    # d = dict((1, 'a'), (2, 'b'))  # ещё вариант создания
    # d[3] = 'c'  # добавить или заменить элемент словаря
    # value = d.get(3, None)  # получить (get) и удалить (pop) элемент словаря, а если его нет, то вернуть значение по умолчанию (в данном случае - None)
    # for k, v in d.items()    for k in d.keys() (или просто "in d")    for v in d.values() - варианты прохода по словарю

    # if 6 in l:  # проверка на вхождение в list, tuple, set, dict
    #     pass
    # else:
    #     pass

    # print(f'Какое-то число: {1.23}. \nОкруглить до сотых: {1.2345:.2f}. \nВывести переменную: {args.input_file}. \nВывести список: {[1, 2, "a", "b"]}')  # f-string позволяет создавать строки со значениями переменных
    # print('Вывести текст с чем-нибудь другим в конце вместо перевода строки.', end='1+2=3')
    # print()  # 2 раза перевести строку
    # print()
    # print('  Обрезать пробелы по краям строки и перевести всё в нижний РеГиСтР.   \n\n\n'.strip().lower())

    # import copy
    # tmp = copy.deepcopy(d)  # глубокая, полная копия объекта

    # Можно передавать в функцию сколько угодно параметров, если её объявить так:
    # def func(*args, **kwargs):
    # Тогда args - это list, а kwargs - это dict
    # При вызове func(1, 'b', c, par1=2, par2='d') будет: args = [1, 'b', c], а kwargs = {'par1': 2, 'par2': 'd'}.
    # Можно "раскрывать" списки и словари и подавать их в функции как последовательность параметров: some_func(*[l, i, s, t], **{'d': i, 'c': t})

    # p = pathlib.Path('/home/user/Documents') - создать объект Path
    # p2 = p / 'dir/file.txt' - добавить к нему ещё уровени
    # p.glob('*.png') и p.rglob('*.png') - найти все файлы нужного вида в папке, только в этой папке и рекурсивно; возвращает не list, а generator (выдаёт только по одному элементу за раз), поэтому если хотите получить сразу весь список файлов, то надо обернуть результат в "list(...)".