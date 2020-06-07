import ast
import os
import urllib.request
from bs4 import BeautifulSoup as Soup

from selenium import webdriver


driver = webdriver.Safari()
URL = 'https://www.google.com/search?q=walther&tbm=isch&hl=en&chips=q:james+bond+walther,g_1:james+bond:rHugS8GzHcM%3D,g_1:pierce+brosnan:oEvatrs0CHk%3D&bih=862&biw=1440&client=safari&hl=en&ved=2ahUKEwjd1JqgsafnAhUMOBQKHYW_DugQ4lYoAnoECAEQGw'
Save_to = "/Applications/YOLOv3/Gun_Images"
f = open("images.txt","a")

def get_images_url():

    driver.get(URL)
    a = input()
    page = driver.page_source

    soup = Soup(page, 'lxml')

    keyword = soup.findAll('div', {'class':'rg_meta notranslate'})
    urls = []

    for url in keyword:
        urls.append(ast.literal_eval(url.text)['ou'])

    return urls


def download_images(images):

    for i, image in enumerate(images):
        image_path = os.path.join(Save_to, '{:05}.jpg'.format(i))

        try:
            urllib.request.urlretrieve(image, image_path)
        except:
            print("Failed", i, "most likely because of certificate issue", image)

images_url = get_images_url()
download_images(images_url)
f.close()

