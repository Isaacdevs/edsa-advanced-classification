"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
from lzma import PRESET_DEFAULT
from matplotlib.collections import Collection
from nbformat import write
import streamlit as st
import joblib,os
from PIL import Image
import time

# Data dependencies
import pandas as pd

# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	st.set_page_config(page_title="Data Innov", page_icon=":earth_africa:")

	# Creates a main title and subheader on your page -
	# these are static across all pages
	title_logo, title = st.columns((1,5))

	with title_logo:
		st.image("resources/images/logo1 (1).png")

	with title:
		st.title("""DATA INNOV CLASSIFIERS""")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["About Us", "Prediction", "Information"]
	selection = st.sidebar.radio("Navigation", options, horizontal=False)



	if selection == "About Us":
	
		st.markdown("<h2 style='text-align: left; color: white;'>About The Team</h2>", unsafe_allow_html=True)


		team_img, team_info = st.columns((1,3))

		with team_img:
			st.image('resources/images/team.jpeg')
		with team_info:
			st.write(
				"""
				Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor
				incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis
				nost
				
				"""
			)	


		st.markdown("<h2 style='text-align: center; color: orange;'>About The Platform</h2>", unsafe_allow_html=True)

		# st.write("##")

		project_info, project_img = st.columns((2,1))

		with project_info:
			st.write(
				"""
				Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor
				incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis
				nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. 
				Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore
				 eu fugiat nulla pariatur. Excepteur sint occaeca



				nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. 
				Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore
				 eu fugiat nulla pariatur. Excepteur sint occaeca

				nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. 
				Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore
				 eu fugiat nulla pariatur. Excepteur sint occaeca  
				
				"""
			)		

		with project_img:
			st.image('resources/images/climate-change.jpg')
			st.image('resources/images/climate-change2.jpeg')


	# Building out the predication page
	if selection == "Prediction":
		st.subheader("Climate change tweet classification")
		# Creating a text box for user input




		# st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center;} </style>', unsafe_allow_html=True)

		# st.write('<style>div.st-bf{font-weight:bold;flex-direction:column;} div.st-ag{font-weight:bold;padding-left:10px;}</style>', unsafe_allow_html=True)

		chosen_model =st.radio("",("Logistic Regression", "SVC", "SVM", "KNN" ), horizontal=True)



		tweet_text = st.text_area("Enter Text","Type Here")
		col1, col2, col3 = st.columns([2,1,2])

		if col2.button("CLASSIFY"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
			prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.

			with st.spinner(text='In progress'):
				time.sleep(2)
				st.balloons()
		
			if(int(prediction) == -1):
				st.image('resources/images/img.gif')	
			elif(int(prediction) == 0):
				st.image('resources/images/img0.gif')
			elif(int(prediction) == 1):
				st.image('resources/images/img1.gif')
			else:
				st.image('resources/images/img2.gif')
				

	# Building out the "Information" page
	if selection == "Information":
		# You can read a markdown file from supporting resources folder
		st.write("---")
		st.markdown("<h1 style='text-align: centre; color: orange;'>Data Collection</h1>", unsafe_allow_html=True)

		collection_text, collection_img = st.columns((2,1))

		with collection_text:
			st.write(
				"""
				Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor
				incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis
				nost
				
				"""
			)
		with collection_img:
			st.image('resources/images/kaggle1.jpg')			


		if st.checkbox('View raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page


		
		st.markdown("<h1 style='text-align: right; color: white;'>Data Processing</h1>", unsafe_allow_html=True)

		st.write(
				"""
				Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor
				incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis
				"""
			)

		_image, _text = st.columns((1, 3))

		with _image:
			pass
		with _text:
			pass

	
		st.markdown("<h1 style='text-align: center; color: orange;'>Data Processing</h1>", unsafe_allow_html=True)

		st.write(
				"""
				Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor
				incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis
				"""
			)

		_image, _text = st.columns((1, 3))

		with _image:
			pass
		with _text:
			pass

		
		st.markdown("<h1 style='text-align: centre; color: white;'>Feedback</h1>", unsafe_allow_html=True)


		
		st.markdown("<h2 style='text-align: center; color: orange;'>More information</h2>", unsafe_allow_html=True)

		_image, _text = st.columns((1, 1))

		with _image:
			st.markdown("<h4 style='text-align: left; color: white;'>Information links</h4>", unsafe_allow_html=True)
		with _text:	
			st.markdown("<h4 style='text-align: left; color: white;'>Recomended Tutorials</h4>", unsafe_allow_html=True)








# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
