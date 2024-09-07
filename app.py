




import os
import json
from flask import Flask, request, Response, stream_with_context, jsonify, send_from_directory
from openai import OpenAI
from flask_cors import CORS
from dotenv import load_dotenv
import boto3
from botocore.exceptions import BotoCoreError, ClientError
from contextlib import closing
import uuid
import logging

load_dotenv()

app = Flask(__name__)
CORS(app)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = OpenAI(api_key=os.environ.get("OPEN_AI_API_KEY"))
aws_access_key_id = os.environ.get("AWS_ACCESS_KEY")
aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
aws_region = os.environ.get("AWS_REGION", "us-east-1")

polly_client = boto3.client('polly',
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=aws_region
)

# Create audio_files directory if it doesn't exist
AUDIO_FILES_DIR = os.path.join(os.path.dirname(__file__), 'audio_files')
os.makedirs(AUDIO_FILES_DIR, exist_ok=True)

streaming = False
stream_state = {
    "chapter": 0,
    "chunk": 0
}
rearranged_chapters = []
user_data = {}

@app.route('/stop_streaming', methods=['POST'])
def stop_streaming():
    global streaming
    streaming = False
    return jsonify({"message": "Streaming stopped"}), 200

@app.route('/addinfo', methods=['POST'])
def add_info():
    global user_data
    data = request.json
    user_data.update(data)
    return jsonify({"message": "User data updated"}), 200

@app.route('/asknextquestion', methods=['POST'])
def ask_next_question():
    data = request.json
    answer = data.get("answer")
    question_number = data.get("questionNumber", 0)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[
            {"role": "system", "content": f"You are a personalized coach for the book '12 Rules for Life'. This is question number {question_number + 1} out of 3. Based on the user's answer, ask a follow-up question to understand them better."},
            {"role": "user", "content": f"User's answer: {answer}. Ask the next question."}
        ]
    )

    return response.choices[0].message.content


@app.route('/audio/<filename>')
def serve_audio(filename):
    return send_from_directory(AUDIO_FILES_DIR, filename, as_attachment=True)


def read_chapter_content(chapter_name):
    filename = os.path.join(os.path.dirname(__file__), 'chapters', f"{chapter_name}.txt")
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            content = file.read().strip()
            return {"name": f"Chapter {chapter_name}", "content": content}
    except FileNotFoundError:
        logger.warning(f"Chapter file not found: {filename}")
        return {"name": f"Chapter {chapter_name}", "content": "Content not available."}
    except Exception as e:
        logger.error(f"Error reading chapter {chapter_name}: {str(e)}")
        return {"name": f"Chapter {chapter_name}", "content": "Error reading chapter content."}

def get_all_chapters():
    chapters = []
    for i in range(1, 13):  # Assuming 12 chapters
        chapter = read_chapter_content(str(i))
        chapters.append(chapter)
    return chapters

def rearrange_chapters(qa):
    chapters = get_all_chapters()
    
    chapter_info = "\n".join([f"{i+1}. {c['name']}: {c['content'][:100]}..." for i, c in enumerate(chapters)])
    
    prompt = f"""Given the following information about a person:
Q1: {qa['q1']}
A1: {qa['a1']}
Q2: {qa['q2']}
A2: {qa['a2']}
Q3: {qa['q3']}
A3: {qa['a3']}

And the following book chapters from "12 Rules for Life: An Antidote to Chaos" by Jordan B. Peterson:

{chapter_info}

Rearrange all the 12 chapter numbers in an order that would be most beneficial for this person, considering their responses. Only provide the rearranged all 12 numbers, separated by commas:"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[
            {"role": "system", "content": "You are an AI assistant that rearranges all the 12 chapters based on user responses. All the chapters should be present on the basis of user responses"},
            {"role": "user", "content": prompt}
        ]
    )

    rearranged_indices = [int(i.strip()) for i in response.choices[0].message.content.split(',') if i.strip().isdigit()]
    return [chapters[i-1] for i in rearranged_indices if 1 <= i <= len(chapters)]

def synthesize_speech(text):
    try:
        response = polly_client.synthesize_speech(
            Text=text,
            OutputFormat="mp3",
            VoiceId="Joanna"
        )
        
        if "AudioStream" in response:
            filename = f"speech_{uuid.uuid4()}.mp3"
            file_path = os.path.join(AUDIO_FILES_DIR, filename)
            try:
                with closing(response["AudioStream"]) as stream:
                    with open(file_path, "wb") as file:
                        file.write(stream.read())
                return f"/audio/{filename}"
            except IOError as error:
                logger.error(f"Error writing audio file: {error}")
                return None
    except (BotoCoreError, ClientError) as error:
        logger.error(f"Error in speech synthesis: {error}")
        return None

def modify_chapter_content(chapter, qa):
    chunk_size = 500
    global user_data
    chunks = [chapter['content'][i:i+chunk_size] for i in range(0, len(chapter['content']), chunk_size)]

    for i, chunk in enumerate(chunks):
        user_data_str = "\n".join([f"{key}: {value}" for key, value in user_data.items()])
        prompt = f"""Given the following information about a person:
Q1: {qa['q1']}
A1: {qa['a1']}
Q2: {qa['q2']}
A2: {qa['a2']}
Q3: {qa['q3']}
A3: {qa['a3']}
Additional user information:
{user_data_str}

This is part {i+1} of {len(chunks)} of the chapter content.
Modify the following chapter content to make it more relevant and personalized for this individual:

{chunk}

Please provide the modified content, maintaining the overall structure and flow of the original text:"""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": "You are an AI assistant that modifies book chapters to make them more personalized and relevant based on user responses. Maintain the overall structure and flow of the original text."},
                {"role": "user", "content": prompt}
            ]
        )

        modified_content = response.choices[0].message.content
        
        # Generate audio and yield the URL immediately
        audio_url = synthesize_speech(modified_content)
        if audio_url:
            yield {
                "name": chapter['name'],
                "content": None,
                "audio_url": audio_url,
                "part": i+1,
                "total_parts": len(chunks)
            }

        # Then yield the modified content
        yield {
            "name": chapter['name'],
            "content": modified_content,
            "audio_url": None,
            "part": i+1,
            "total_parts": len(chunks)
        }

@app.route('/stream_chapters', methods=['POST'])
def stream_chapters():
    global streaming, stream_state, rearranged_chapters
    data = request.json
    
    if not rearranged_chapters or data.get('rearrange', False):
        rearranged_chapters = rearrange_chapters(data)
    
    streaming = True
    
    logger.info(f"Streaming set to {streaming} at start of stream_chapters")
    logger.info(f"Initial stream_state: {stream_state}")
    
    def generate():
        global streaming, stream_state
        
        try:
            for chapter_index in range(stream_state["chapter"], len(rearranged_chapters)):
                if not streaming:
                    logger.info("Streaming stopped. Breaking out of chapter loop.")
                    break
                
                chapter = rearranged_chapters[chapter_index]
                
                for modified_chunk in modify_chapter_content(chapter, data):
                    if not streaming:
                        logger.info("Streaming stopped. Breaking out of chunk loop.")
                        break
                    
                    stream_state["chunk"] += 1
                    logger.info(f"Yielding chunk. Chapter: {chapter_index}, Chunk: {stream_state['chunk']}")
                    yield f"data: {json.dumps(modified_chunk)}\n\n"
                
                if streaming:
                    stream_state["chapter"] += 1
                    stream_state["chunk"] = 0  # Reset chunk index for the next chapter
                    logger.info(f"Finished chapter {chapter_index}. New state: {stream_state}")
            
            if streaming:
                logger.info("Streaming completed successfully")
                stream_state = {"chapter": 0, "chunk": 0}  # Reset state when finished
                yield "data: {\"done\": true}\n\n"
            else:
                logger.info(f"Streaming was interrupted. Final state: {stream_state}")
                yield f"data: {{\"interrupted\": true, \"state\": {json.dumps(stream_state)}}}\n\n"
        
        except Exception as e:
            logger.error(f"Error during streaming: {str(e)}")
            yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"
        
        finally:
            logger.info(f"Streaming ended. Final state: streaming={streaming}, stream_state={stream_state}")

    return Response(stream_with_context(generate()), mimetype='text/event-stream')


@app.route('/ask_question', methods=['POST'])
def ask_question():
    data = request.json
    question = data.get('question')

    if not question:
        return jsonify({"error": "No question provided"}), 400

    prompt = f"""You are Jordan B. Peterson, the author of "12 Rules for Life: An Antidote to Chaos". A reader has the following question:

Question: {question}

Please answer the question as Jordan B. Peterson would, based on your expertise and the ideas presented in your book. Keep the answer concise but informative."""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[
            {"role": "system", "content": "You are Jordan B. Peterson, answering questions about your book '12 Rules for Life: An Antidote to Chaos'."},
            {"role": "user", "content": prompt}
        ]
    )

    return jsonify({"answer": response.choices[0].message.content})



if __name__ == '__main__':
    app.run(debug=True)
