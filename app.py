import streamlit as st
from dotenv import load_dotenv
from utils import *
import uuid

#Creating session variables
# if 'unique_id' not in st.session_state:
#     st.session_state['unique_id'] =''
st.session_state['unique_id'] = "aaa365fe031e4b5ab90aba54eaf6012e"

def main():
    load_dotenv()

    st.set_page_config(page_title="Resume Screening Assistance")
    st.title("HR - Resume Screening Assistance...💁 ")
    st.subheader("I can help you in resume screening process")

    job_description = st.text_area("Please paste the 'JOB DESCRIPTION' here...",key="1")
    document_count = st.text_input("No.of 'RESUMES' to return",key="2")
    # Upload the Resumes (pdf files)
    pdf = st.file_uploader("Upload resumes here, only PDF files allowed", type=["pdf"],accept_multiple_files=True)

    submit=st.button("Help me with the analysis")

    if submit:
        with st.spinner('Wait for it...'):

            #Creating a unique ID, so that we can use to query and get only the user uploaded documents from PINECONE vector store
            st.session_state['unique_id']=uuid.uuid4().hex

            #Create a documents list out of all the user uploaded pdf files
            final_docs_list=create_docs(pdf,st.session_state['unique_id'])

            #Displaying the count of resumes that have been uploaded
            st.write("*Resumes uploaded* :"+str(len(final_docs_list)))

            #Create embeddings instance
            embeddings=create_embeddings_load_data()

            #Push data to PINECONE
            push_to_pinecone("ad12a7c3-b36f-4b48-986e-5157cca233ef","gcp-starter","resume-db",embeddings,final_docs_list)

            #Fecth relavant documents from PINECONE
            # relavant_docs=similar_docs(job_description,document_count,"ad12a7c3-b36f-4b48-986e-5157cca233ef","gcp-starter","resume-db",embeddings,st.session_state['unique_id'])

            relavant_docs=similar_docs(job_description,document_count,"ad12a7c3-b36f-4b48-986e-5157cca233ef","gcp-starter","resume-db",embeddings,st.session_state['unique_id'])            #t.write(relavant_docs)

            #Introducing a line separator
            st.write(":heavy_minus_sign:" * 30)

            #For each item in relavant docs - we are displaying some info of it on the UI
            st.session_state['counter'] = index =  0 
            
            if st.button('Increment Counter'):
    # Increment the counter when the button is clicked
                st.session_state['counter'] += 1
                
                index =  st.session_state['counter']
   
            if st.button('Decrement Counter'):
                    if button_state > 1 : 
                        st.session_state['counter'] -= 1
                        index =  st.session_state['counter']

            
            #For each item in relavant docs - we are displaying some info of it on the UI
            for item in range(len(relavant_docs)):
                
                st.subheader("👉 "+str(item+1))

                #Displaying Filepath
            
                # for item in range(len(relavant_docs)):
                    
                #Displaying File Name 
                st.write("**File** : "+relavant_docs[item][0].metadata['name'])
#metadata_filename( relevant_docs[item][0] ) 
    
                #Introducing Expander feature
                with st.expander('Show me 👀'): 
                    st.info("**Match Score** : "+str(relavant_docs[index][1]))
                    st.write("***"+relavant_docs[item][0].page_content)
    
                    #Gets the summary of the current item using 'get_summary' function that we have created which uses LLM & Langchain chain
                    summary = get_summary(relavant_docs[item][0]) 
                    st.write("**Summary** : "+summary)

        st.success("Hope I was able to save your time❤️")


#Invoking main function
if __name__ == '__main__':
    main()
