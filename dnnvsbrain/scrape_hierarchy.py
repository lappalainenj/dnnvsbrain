import requests
from bs4 import BeautifulSoup
import time
from selenium import webdriver
from .experiment import json_path
from urllib.parse import quote
import re
from tqdm import tqdm
import json

with open(json_path / 'labels.json', 'r') as f:
    labels = json.load(f)

def get_url(label):
    """Get imagenet search url for the label."""
    url = 'http://www.image-net.org/search?q='
    return url + quote(label)

def get_wnid(label):
    """Obtain the wnid from requesting a search."""
    if label == 'teddy, teddy bear':
        return 'n04399382'
    r = requests.get(get_url(label))
    soup = BeautifulSoup(r.content, "lxml")
    AA = soup.findAll('a', attrs = {'href':re.compile('synset\?wnid=.+')})
    wnids = [A['href'].replace('synset?wnid=', '') for A in AA if A.find('span')]
    synsets = [A.find('span').get_text().replace('Synset: ', '') for A in AA if A.find('span')]
    idx = [i for i, x in enumerate(synsets) if x == label][0]
    return wnids[idx]

def render_page(url):
    """Render javascript with Chrome webdriver"""
    options = webdriver.ChromeOptions()
    options.headless = True
    driver = webdriver.Chrome(options = options, executable_path='/home/j.lappalainen/chromedriver_linux64/chromedriver')
    try:
        driver.get(url)    
    except Exception as e:
        print(e + ' ...trying again.')
        time.sleep(5)
        render_page(url)
    time.sleep(3)
    r = driver.page_source
    driver.quit()
    return r

def get_hierarchy(label):
    """Get the hierarchy from imagenet treeview for a label"""
    wnid = get_wnid(label)
    url = 'http://www.image-net.org/synset?wnid='+wnid 
    r = render_page(url)
    soup = BeautifulSoup(r, "lxml")
    soup_hierarchy = soup.findAll('a', attrs={"class":"jstree-my-clicked"})
    hierarchy = [re.sub(" \(.+\)", '', sh.contents[1]) for sh in soup_hierarchy]
    idx = [i for i, x in enumerate(hierarchy) if x == label][0]
    return hierarchy[1:idx+1]

if __name__=='__main__':
    hierarchy = {}
    for key, label in tqdm(list(labels.items())):
        hierarchy[label] = get_hierarchy(label)

    with open(json_path + '/class_hierarchy.json', 'w') as fp:
        json.dump(hierarchy, fp)
    