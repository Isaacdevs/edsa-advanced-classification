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
import streamlit as st
import joblib,os
from PIL import Image

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
	selection = st.sidebar.radio("Navigation", options)



	if selection == "About Us":
		st.write("---")
		st.header("About the Team")
		st.write("##")

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


		st.header("About the Plaform")
		st.write("##")

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




	# Building out the "Information" page
	if selection == "Information":
		# You can read a markdown file from supporting resources folder
		st.markdown("Some information here")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

	# Building out the predication page
	if selection == "Prediction":
		st.subheader("Climate change tweet classification")
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")

		if st.button("Classify"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
			prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			st.snow()
			st.success("Text Categorized as: {}".format(prediction))

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
