# Use an official Python runtime as a parent image
FROM python:3.9

# Install system dependencies
RUN apt-get update

# Set the working directory in the container
WORKDIR /app

# Install special lib
COPY pgpelib/ pgpelib/

# Install the lib
RUN pip install pgpelib/

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the notebook into the container
COPY test.ipynb .

# Copy the data folder into the container
COPY  data/ data/
COPY  utils/ utils/

# Expose the port Jupyter will run on
EXPOSE 8888

# Run Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
