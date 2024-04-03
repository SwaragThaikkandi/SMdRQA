import openai
import os
import requests
import json

def get_openai_response(prompt):
    openai.api_key = os.getenv('OPENAI_API_KEY')
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=200,
        temperature=0.7
    )
    return response.choices[0].text.strip()

def post_github_comment(comment, pull_request_number):
    github_token = os.getenv('GITHUB_TOKEN')
    repo_name = os.getenv('GITHUB_REPOSITORY')
    url = f'https://api.github.com/repos/{repo_name}/issues/{pull_request_number}/comments'
    headers = {'Authorization': f'token {github_token}', 'Accept': 'application/vnd.github.v3+json'}
    data = {'body': comment}
    response = requests.post(url, headers=headers, json=data)
    return response.status_code

if __name__ == '__main__':
    prompt = "Please review the following code and provide feedback:"
    openai_response = get_openai_response(prompt)

    pull_request_number = os.getenv('PR_NUMBER')  # Assuming you pass the PR number as an environment variable
    comment = f"**GPT-3.5 Code Review:**\n{openai_response}"
    status_code = post_github_comment(comment, pull_request_number)

    if status_code == 201:
        print('Comment posted successfully!')
    else:
        print('Failed to post comment to GitHub.')
