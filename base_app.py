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
from sqlalchemy import column
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
	options = ["Predictions", "About Us" , "Information"]
	selection = st.sidebar.radio("Navigation", options, horizontal=False)



	if selection == "About Us":
	
		st.markdown("<h2 style='text-align: left; color: white;'>About The Team</h2>", unsafe_allow_html=True)


		team_img, team_info = st.columns((1,3))

		with team_img:
			st.image('resources/images/team.jpeg')
		with team_info:
			st.write(
				"""
				DATA INNOV: is a team of Data Scientist'. We collect data and take it throught several
				Data Science processes to provide valuable insights. These insights can be in the form
				of Power BI visualisations or Machine learning Models, which is what was used for this
				platform.  
				
				"""
			)	


		st.markdown("<h2 style='text-align: center; color: orange;'>About The Platform</h2>", unsafe_allow_html=True)

		# st.write("##")

		project_info, project_img = st.columns((2,1))

		with project_info:
			st.write(
				"""

				Many companies are built around lessening one’s environmental impact or carbon footprint. They
				offer products and services that are environmentally friendly and sustainable, in line with 
				their values and ideals. They would like to determine how people perceive climate change and 
				whether or not they believe it is a real threat. This would add to their market research efforts
				in gauging how their product/service may be received.


				This platform provides an accurate and robust solution to give companies access to a broad
				base of consumer sentiment, spanning multiple demographic and geographic categories - 
				thus increasing their insights and informing future marketing strategies.

				There are two more section in addition About Us namely predictions and Information.
				Predictions page for classifying text using different Models and the information page 
				where we give some details about the processes we went through, a feedback form and 
				some links for more information.

				"""
			)		

		with project_img:
			st.image('resources/images/climate-change.jpg')
			st.image('resources/images/climate-change2.jpeg')


	# Building out the predication page
	if selection == "Predictions":
		st.subheader("Climate change tweet classification")
		# Creating a text box for user input




		# st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center;} </style>', unsafe_allow_html=True)

		# st.write('<style>div.st-bf{font-weight:bold;flex-direction:column;} div.st-ag{font-weight:bold;padding-left:10px;}</style>', unsafe_allow_html=True)

		chosen_model =st.radio("",("Logistic_Regression", "SVC", "SVM", "KNN" ), horizontal=True)



		tweet_text = st.text_area("Enter Text","Type Here")
		col1, col2, col3 = st.columns([2,1,2])

		if col2.button("CLASSIFY"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join("resources/{0}.pkl".format(chosen_model)),"rb"))
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
				Data The collection of this data was funded by a Canada Foundation for Innovation JELF Grant to Chris Bauch,
				University of Waterloo. The dataset aggregates tweets pertaining to climate change collected between Apr 27,
				2015 and Feb 21, 2018. In total, 43943 tweets were collected. Each tweet is labelled as one of the following
				classes:
				
				"""
			)
		with collection_img:
			st.image('resources/images/kaggle1.jpg')			


		if st.checkbox('View raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

		
		st.markdown("<h1 style='text-align: right; color: white;'>Data Processing</h1>", unsafe_allow_html=True)

		st.write(
				"""
				After collecting the data, we went through several stages to process it using some data handling techniques 
				found in Data Sciences.
				"""
			)

		data_clean1, data_clean2 = st.columns((1, 1))
		with data_clean1:
			st.image('resources/images/cleaning_data.gif')
		with data_clean2:
			st.markdown("<h2 style='text-align: center; color: black;'>Data Cleaning</h2>", unsafe_allow_html=True)
			st.write("""We performed some data cleaning where we reomved unnecesary things.
						In our case, we removed all the puctuation and URL's which are very common in tweets. Lastly we
						removed all the english stopword which are simply commonly used words.
		
					""")	
			


		st.markdown("<h2 style='text-align: center; color: orange;'>Data Balancing</h2>", unsafe_allow_html=True)
		st.write("""
				One of the most interesting observations we found was the imbalances that existed in the data.
				This is very significant in classification because it might may cause overfitting. We used a 
				sampling method to balance the data as illustrated in the following figures.""")	
		data_image1,space, data_image2 = st.columns((3, 1, 3))
		with data_image1:
			st.image('resources/images/data_imbalace.png')
		with data_image2:
			st.image('resources/images/data_balance.png')
		st.write("\n\n")
		feat_eng1, feat_eng2 = st.columns((2,1))	
		with feat_eng1:
			st.markdown("""
			##### We also ran the following processes:
			- Tokenization
			- Lemmatization
			- Stemming
			- N-grams
			- Vectorization (Tf-IDF)
			""")
		with feat_eng2:
			st.image('resources/images/gears1.gif')
			st.markdown("<h3 style='text-align: center; color: black;'>Feature Engineering</h3>", unsafe_allow_html=True)
			




		st.markdown("Running these ensured that the data is correctly formatted and ready for model training the models.")



	
		st.markdown("<h1 style='text-align: center; color: orange;'>Models</h1>", unsafe_allow_html=True)
		
		_text, _image = st.columns((3, 1))
		with _text:	
			st.write(
					"""
					We discuss the Machine Learning models that were used with a brief explanation of each model.
					Then we do a visial model comparison where we compare the models using the observation we saw
					in the model evaluation or validation processes.
					"""
				)
		with _image:
			st.image('resources/images/climate-change2.png')

		# st.markdown("<h2 style='text-align: center; color: white;'>Models Used</h2>", unsafe_allow_html=True)	

		model1, model2, model3 = st.columns((1,1,1))

		with model1:
			st.markdown("<h3 style='text-align: center; color: orange;'>Logistic Regression</h3>", unsafe_allow_html=True)
			st.write("Logistic Regression was used in the biological sciences in early twentieth century. It was then used in many social science applications. Logistic Regression is used when the dependent variable(target) is categorical.")

		with model2:
			st.markdown("<h3 style='text-align: center; color: white;'>Support Vector Machines</h3>", unsafe_allow_html=True)
			st.markdown("<p style='color: black'>The objective of the support vector machine algorithm is to find a hyperplane in an N-dimensional space(N — the number of features) that distinctly classifies the data points.</p>", unsafe_allow_html=True)
		with model3:
			st.markdown("<h3 style='text-align: center; color: orange;'>K Nearest Neighbours</h3>", unsafe_allow_html=True)
			st.write("The k-nearest neighbors (KNN) algorithm is a simple, easy-to-implement supervised machine learning algorithm that can be used to solve both classification and regression problems.")


		st.markdown("<h2 style='text-align: center; color: white;'>Model Comparison</h2>", unsafe_allow_html=True)	
		st.image('resources/images/data_model_comp1.png')

		st.markdown("<h2 style='text-align: center; color: orange;'>Best and Worst Performing Model</h2>", unsafe_allow_html=True)	
		model_perf1, model_perf2 = st.columns((1, 1))

		with model_perf1:
			st.markdown("<h3 style='text-align: center; color: black;'>Support Vector Machine</h3>", unsafe_allow_html=True)
			st.image('resources/images/st_SVM.png')
		with model_perf2:
			st.image('resources/images/st_RF.png')
			st.markdown("<h3 style='text-align: center; color: orange;'>Random Forest Classifier</h3>", unsafe_allow_html=True)

		
		st.markdown("<h2 style='text-align: centre; color: white;'>Feedback & Recommendations</h2>", unsafe_allow_html=True)

		contact_form ="""
			<form action="https://formsubmit.co/datainnov4@gmail.com" method="POST">
    			 <input type="text" name="Message" placeholder="Your Message" required>
        		<input type="email" name="email" placeholder="Your email" required>
     			<button type="submit">Send</button>
				</form>
			"""

		local_css()
		st.markdown(contact_form, unsafe_allow_html=True)



		st.markdown("<h2 style='text-align: center; color: orange;'>More information</h2>", unsafe_allow_html=True)

		_plug, _image, _text = st.columns((2,2, 2))

		with _plug:
			st.markdown("<h4 style='text-align: center; color: orange;'>We're the</h4>", unsafe_allow_html=True)
			st.image('resources/images/plug.jpg')
			st.markdown("<h2 style='text-align: center; color: orange;'>Plug!</h2>", unsafe_allow_html=True)

		with _image:
			st.markdown("<h4 style='text-align: left; color: white;'>Information links</h4>", unsafe_allow_html=True)
			st.write("[Climate Change](https://www.un.org/en/climatechange/what-is-climate-change)")
			st.write("[Kaggle Competitions](https://www.kaggle.com/competitions)")
			st.write("[Machine Learning](https://www.hpe.com/za/en/what-is/machine-learning.html?jumpid=ps_1ujsi45bwk_aid-520061736&ef_id=Cj0KCQjw8O-VBhCpARIsACMvVLOtzvOcyYaTZ3pQVhF7p_XCUOjJDHXYRLkA71JsJOnsJEFbTVvyn9QaAomCEALw_wcB:G:s&s_kwcid=AL!13472!3!595096975661!e!!g!!what%20is%20machine%20learning!17059482849!137455657433&)")
			st.write("[Streamlit](https://streamlit.io/)")
			st.write("[AWS hosting](https://aws.amazon.com/)")

		with _text:	
			st.markdown("<h4 style='text-align: left; color: orange;'>Recomended Tutorials</h4>", unsafe_allow_html=True)
			st.write("[W3 Schools](https://aws.amazon.com/)")
			st.write("[Machine learning Youtube](https://aws.amazon.com/)")
			st.write("[DataCamp](https://aws.amazon.com/)")


def local_css():
	with open('styles.css') as f:
		st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
