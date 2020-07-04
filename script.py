from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import pandas as pd
import bs4
from bs4 import BeautifulSoup
from googlesearch import search 
import requests
from urllib.request import urlopen
import re
import flask
from flask import Flask, render_template, request, redirect, url_for
from new import modelloading
from new import image_fetch
from new import recommend

app = flask.Flask(__name__,template_folder='template')
@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
	modelloading()
	query = request.form['moviename']
	testing=image_fetch(query)
	movielist=recommend(query)
	if isinstance(testing,list):
		genre=','.join(testing)
		return render_template('index.html',prediction_text="{} : {}".format(query,genre),heading="Recommended Movies's :",link1=movielist[0][1],movie1=movielist[0][0],link2=movielist[1][1],movie2=movielist[1][0],link3=movielist[2][1],movie3=movielist[2][0],link4=movielist[3][1],movie4=movielist[3][0],link5=movielist[4][1],movie5=movielist[4][0])
	else:
		genre=testing
		return render_template('index.html',prediction_text=genre,heading="",movie1="",movie2="",movie3="",movie4="",movie5="",link1="",link2="",link3="",link4="",link5="")
		
if __name__ == '__main__':
	app.run(debug=True)