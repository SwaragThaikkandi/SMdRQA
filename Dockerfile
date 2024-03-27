# Info of the current SMdRQA docker image:
## Modified by Your Name, Your Institution, Your Location.
## Please contact youremail@example.com if you have any question with the current SMdRQA image

# Use the Jupyter minimal notebook as the base image with Python 3.8.8
FROM jupyter/minimal-notebook:python-3.8.8

# Metadata
LABEL maintainer="Your Name <youremail@example.com>"

# Switch to root user for installing system packages
USER root

# Install system packages required by your project
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Switch back to the default non-root user defined by the base image
USER $NB_UID

# Copy requirements.txt from your project directory to the Docker image
COPY requirements.txt /home/${NB_USER}/requirements.txt

# Install Python packages from requirements.txt
RUN pip install --no-cache-dir -r /home/${NB_USER}/requirements.txt && \
    fix-permissions "/home/${NB_USER}"

# Copy any additional files or scripts needed for your project
# Example: COPY /path/to/your/files /home/${NB_USER}/your_files

# Set environment variables if needed
# ENV EXAMPLE_VAR=example_value

# Expose the port number if your project requires it
# EXPOSE 8888

# Start the notebook server or any other command to run your project
# CMD ["jupyter", "notebook", "--ip", "0.0.0.0", "--port", "8888", "--no-browser"]

