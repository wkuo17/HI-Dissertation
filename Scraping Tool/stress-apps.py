import time
from time import sleep
from bs4 import BeautifulSoup
import pandas as pd
import sys, io
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.proxy import *
import csv
import timeit
from multiprocessing import Pool
#number of reviews - initial 40 + multiples of 40. (eg. 0 -> 40 + 40(0) = 40, 10 -> 40 + 40(10) = 440)
multiples_of_40 = 100

driver = webdriver.Firefox(executable_path='/usr/local/bin/geckodriver') #Initiate driver

#create a urls array to store app store urls
urls = []

#"stress" apps
driver.get("https://play.google.com/store/search?q=stress&c=apps")

#loop the number of times we want to fetch urls
for i in range(20):
    try:
        show_more_button = driver.find_element_by_id("show-more-button")
        show_more_button.click()
        sleep(3)

    except Exception:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        sleep(3)


page = driver.page_source
soup_expatistan = BeautifulSoup(page, "html.parser")

for i in soup_expatistan.find_all('div', attrs={'class' : 'cover'}):
    ##print (i.a['href'])
    urls.append("https://play.google.com" + i.a['href'] + "&showAllReviews=true")

download_dir = "stress-apps.csv" #where you want the file to be downloaded to 

csv_file = open(download_dir, "w") #"w" indicates that you're writing strings to the file
csv_writer=csv.writer(csv_file) #create writer

#column names for title row
csv_writer.writerow(("appname", "rating", "ratingcount", "developer", "apptype", "reviewer", "date", "reviewer_rating", "thumbsup", "review"))

start = timeit.default_timer()


for url in urls:
    try:
        show_more_button_available = False
        count = 0
        driver.get(url)

                #loop the number of times we want to fetch reviews
        for i in range(multiples_of_40):
            try:
                #if the "Show More" button is present, then attempt to click on it to load more reviews
                show_more_button = driver.find_element_by_class_name("RveJvd")
                show_more_button.click()
                ##print ("show more button")
                show_more_button_available = False
                count = 0
                
            except Exception:
                ##print ("exception scroll to ")
                ##print (show_more_button_available)
                if (show_more_button_available) and (count > 3):
                    break
                SCROLL_PAUSE_TIME = 1

                # Get scroll height
                last_height = driver.execute_script("return document.body.scrollHeight")

                while True:
                  # Scroll down to bottom
                    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

                 # Wait to load page
                    sleep(SCROLL_PAUSE_TIME)

                  # Calculate new scroll height and compare with last scroll height
                    new_height = driver.execute_script("return document.body.scrollHeight")
                    #print (new_height)
                  
                    if new_height == last_height:
                        show_more_button_available = True
                        count += 1
                        break
                    last_height = new_height

        page = driver.page_source #access page source

        soup_expatistan = BeautifulSoup(page, "html.parser") #get page source html code

        #app info
        appname = str(soup_expatistan.find("h1", class_="AHFaub").span.string)
        rating = str(soup_expatistan.find("div", class_="pf5lIe").next_element.attrs["aria-label"].split(" ")[1])    
        ratingcount = str(soup_expatistan.find("span", class_="AYi5wd TBRnV").text)  
        developer = str(soup_expatistan.find("a", class_="hrTbp R8zArc").text)   
        apptype = str(soup_expatistan.find("a", itemprop="genre").text)

        #User review box
        page = driver.page_source #access page source
        soup_expatistan = BeautifulSoup(page, "html.parser") #get page source html code
        expand_pages = soup_expatistan.find_all("div", class_="d15Mdf bAhLNe")

        print ("Rating - " + url)
        # iterate through pages
        for count, expand_page in enumerate(expand_pages):
            reviewer = expand_page.find("span", class_="X43Kjb").string        
            date = expand_page.find("span", class_="p2TkOb").string  
            reviewer_rating = str(expand_page.find("div", class_="pf5lIe").next_element.attrs["aria-label"].split(" ")[1])
            thumbsup = str(expand_page.find("div", class_="jUL89d y92BAb").text)       
            fullreview = expand_page.find("button", jsname="gxjVle")
            if fullreview is not None:
                review = expand_page.find("span", jsname="fbQN7e").text
                review = "Review Text %s: %s" %(count+1, review)
                review = review.replace("\n", " ")
            else:
                review = expand_page.find("span", jsname="bN97Pc").text
                review = "Review Text %s: %s" %(count+1, review)
                review = review.replace("\n", " ")

            # write rows into csv file
            csv_writer.writerow([appname, rating, ratingcount, developer, apptype, reviewer, date, reviewer_rating, thumbsup, review])
    except Exception:
        driver.get(url) #get URLs
        page = driver.page_source #access page source

        soup_expatistan = BeautifulSoup(page, "html.parser") #get page source html code

        # app info
        appname = str(soup_expatistan.find("h1", class_="AHFaub").span.string)
        print ("No Rating - " + url)
        csv_writer.writerow([appname])
        
    csv_writer.writerow([])

stop = timeit.default_timer()
print (stop - start) 
driver.quit()

