#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      MuskanMishty
#
# Created:     03-08-2016
# Copyright:   (c) MuskanMishty 2016
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import tweepy
import numpy
from tweepy import OAuthHandler
def main():
    pass

if __name__ == '__main__':
    main()

consumer_key = 'f0JTlAHFHCRIGuRA612mx7Xnk'
consumer_secret = 'Kr8xDFWCi0ZbWlokZctv9sFl9KnYub9eawAndiwZmlMLxKOqZq'
access_token = '3228032934-eVK1G38ImBRymJJu1OlxU5PMu8KRrjBdLwyCd6p'
access_secret = '1g99vzRrVwNAsZRyRuLQ0oGcJjM9pLnpjI16YbPhCLqTz'

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)

api = tweepy.API(auth)

i=1

tweetidcorpusone=dict()
with open("tweet_ids", "r") as f:
    for line in f:
        sent=line.split(" ")
        tweetidcorpusone[i]=sent[0]
        i=i+1;
print tweetidcorpusone

countnotfound=0
i=1
tweetcorpusone=dict()
for k,v in tweetidcorpusone.iteritems():
    try:
        if k>1000:
            break
        else:
            tweet=api.get_status(tweetidcorpusone[k])
            tweetcorpusone[i]= tweet.text
            i=i+1

    except:
        countnotfound=countnotfound+1


print tweetcorpusone
print countnotfound