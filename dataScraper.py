from bs4 import BeautifulSoup
import urllib2
import numpy
import csv
import pandas as pd
import time

lyricsvector = []  # input (bag of words)
genrevector = []  # target
songinfovector = []  # metadata (artist and songname)
yearvector = [] # years

# List the URLs here
start = "http://www.songlyrics.com/news/top-songs/"

output = open('lyrics_year.csv', 'a+')

# output.write("songinfo,genre,lyrics,year\n")

# convert top 100 urls into parseable html
for i in range(2007,2010):
    url = start + str(i)
    headers = {'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:54.0) Gecko/20100101 Firefox/54.0',
               'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'}
    req = urllib2.Request(url, None, headers)
    doc = html = urllib2.urlopen(req).read()
    soup = BeautifulSoup(doc, 'html.parser')
    div = soup.find('div', {'class': 'box listbox'})

    # get genres
    title = soup.title.get_text().encode('ascii', 'ignore').split(' ')
    index100 = title.index('100')
    indexSongs = title.index('Songs')
    genre = ' '.join(title[(index100 + 1):(indexSongs)]).encode('utf-8')

    # create list of top 100 songs by genre
    print genre
    songs = div.find_all('a')
    songlinks = []

    # create loop to extract song links
    for j in range(0, 200):  # [0::2]:
        songlink = songs[j].get('href').encode('ascii', 'ignore')
        songlinks.append(songlink)  # output links to a list called songlinks

    songlinks = filter(None, songlinks)
    songlinks = [songlink for songlink in songlinks if (len(songlink.split('/')) == 6)]



    # loop through songlinks list to get the actual lyrics
    for k in range(0, len(songlinks)):
        headers = {'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:54.0) Gecko/20100101 Firefox/54.0',
                   'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'}
        req = urllib2.Request(songlinks[k], None, headers)
        flag = True
        c_flag = False

        count = 0
        while(flag):
            try:
                songdoc = urllib2.urlopen(req).read()
                flag = False
            except:
                time.sleep(30)
                count += 1
                print "Retrying " + str(count)
            finally:
                if count == 3:
                    flag = False
                    c_flag = True
                    print "c_flag set"

        if c_flag:
            continue
        songsoup = BeautifulSoup(songdoc, 'html.parser')
        songinfo = songsoup.title.get_text().encode('ascii', 'ignore')
        print songinfo, 'is number', k

        songdiv = songsoup.find('div', {'id': 'songLyricsDiv-outer'})
        try:
            lyrics = songdiv.getText().replace("\n", " ").replace("\'", "").replace("\r", " ").encode('utf-8')
        except:
            continue
        output_string = songinfo + ",\"" + lyrics + "\"," + str(i) + "\n"
        output.write(output_string)

output.close()
