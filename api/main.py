from fastapi import FastAPI, File, HTTPException

# Set FastAPI app
app = FastAPI(title='Home Credit Default Risk', 
              description='Get information related to the probability of a client not repaying a loan', 
              version='0.1.0')

@app.get('/')
def read_root():
    return 'Home Credit Default Risk API'