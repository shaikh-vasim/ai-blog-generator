import os
import json
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
import markdown
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from functools import lru_cache
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Optional, Dict, Any
import logging
from tqdm import tqdm
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import re
import random
from crewai import LLM
import time
import hashlib
import gradio as gr
from gradio.components import Markdown, HTML
import webbrowser
from pathlib import Path
from blog_generator.agents import BlogPostGenerator


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("blog_generator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("BlogPostGenerator")

# Load environment variables
load_dotenv()


def create_ui():
    """Create Gradio UI for the blog post generator"""
    generator = BlogPostGenerator()
    
    with gr.Blocks(
    title="AI Blog Post Generator", 
    theme="soft",

     css="""
        .gradio-container { 
            background: #0d1017 !important;
            max-width: 1400px !important;
            margin: 0 auto !important;
        }
        body, .block, .tabs, .tabitem, .row, .column {
            background: #0d1017 !important; 
        }
        .markdown, .html {
            background: white !important;
            padding: 20px !important;
            border-radius: 8px !important;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
        }
        .progress-bar {
            height: 6px !important;
            border-radius: 3px !important;
        }
        .status-message {
            padding: 12px;
            border-radius: 4px;
            margin: 10px 0;
        }
        .success-message {
            background-color: #e6f7ee;
            color: #0d1017;
            border-left: 4px solid #0d1017;
        }
        .error-message {
            background-color: #fde8e8;
            color: #c23030;
            border-left: 4px solid #c23030;
        }
        .warning-message {
            background-color: #fff8e6;
            color: #8a6d3b;
            border-left: 4px solid #8a6d3b;
        }
        .loading-message {
            background-color: #e6f3ff;
            color: #1a73e8;
            border-left: 4px solid #1a73e8;
        }
        .generating {
            animation: pulse 1.5s infinite;
        }
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.6; }
            100% { opacity: 1; }
        }
        """
) as demo:
        gr.Markdown("# 📝 AI Blog Post Generator")
        gr.Markdown("Generate, edit, and manage high-quality technical blog posts using AI")
        
        with gr.Tabs():
            with gr.TabItem("Generate New Post"):
                with gr.Row():
                    with gr.Column(scale=2):
                        # Input parameters
                        topic = gr.Textbox(label="Blog Topic", placeholder="e.g., Artificial Intelligence in Healthcare")
                        focus = gr.Textbox(label="Specific Focus", placeholder="e.g., latest trends, applications in finance")
                        date_range = gr.Dropdown(
                            label="Date Range for Research",
                            choices=["All time", "Last week", "Last month", "Last year", 
                                    "Last 1 month", "Last 2 months", "Last 3 months", 
                                    "Last 4 months", "Last 5 months", "Last 6 months"],
                            value="All time"
                        )
                        tone = gr.Dropdown(
                            label="Writing Tone",
                            choices=["Professional", "Conversational", "Technical", "Informative", "Friendly"],
                            value="Professional"
                        )
                        length = gr.Dropdown(
                            label="Post Length",
                            choices=["Short (500-1000 words)", "Medium (1000-2000 words)", "Long (2000+ words)"],
                            value="Medium (1000-2000 words)"
                        )
                        temperature = gr.Slider(
                            label="Creativity Level",
                            minimum=0.1,
                            maximum=1.0,
                            step=0.1,
                            value=0.7,
                            info="Higher values = more creative, lower values = more factual"
                        )
                        add_toc = gr.Checkbox(label="Add Table of Contents", value=True)
                        seo_optimized = gr.Checkbox(label="SEO Optimized", value=True)
                        
                        generate_btn = gr.Button("Generate Blog Post", variant="primary")
                        
                    with gr.Column(scale=3):
                        # Output tabs
                        with gr.Tabs():
                            with gr.TabItem("Markdown Preview"):
                                md_output = gr.Markdown()
                            with gr.TabItem("HTML Preview"):
                                html_output = gr.HTML()
                            with gr.TabItem("Generated Files"):
                                file_output = gr.Files(label="Download Generated Files")
                        
                        # Additional info - MOVE THIS INSIDE THE TABS
                        with gr.Tabs():
                            with gr.TabItem("Generation Details"):
                                with gr.Accordion("Details", open=False):
                                    info_output = gr.JSON(label="Metadata")
                
                # Examples
                examples = gr.Examples(
                    examples=[
                        ["Generative AI in Healthcare", "diagnostic applications", "Last year", "Professional", "Medium (1000-2000 words)", 0.7, True, True],
                        ["Blockchain Technology", "recent advancements in scalability", "Last 3 months", "Technical", "Long (2000+ words)", 0.5, True, True],
                        ["Python Programming", "best practices for data science", "All time", "Conversational", "Short (500-1000 words)", 0.8, False, True]
                    ],
                    inputs=[topic, focus, date_range, tone, length, temperature, add_toc, seo_optimized],
                    outputs=[md_output, html_output, file_output, info_output],
                    fn=generator.generate_blog_post,
                    cache_examples=False,
                    label="Click any example to load parameters"
                )
            
            with gr.TabItem("Edit Existing Post"):
                with gr.Row():
                    with gr.Column(scale=1):
                        post_selector = gr.Dropdown(
                            label="Select Post to Edit",
                            choices=generator.list_generated_posts(),
                            interactive=True
                        )
                        refresh_btn = gr.Button("Refresh List", variant="secondary")
                        load_btn = gr.Button("Load Post", variant="primary")
                        delete_btn = gr.Button("Delete Post", variant="stop")
                        
                        with gr.Accordion("Post Actions", open=True):
                            save_btn = gr.Button("Save Changes", variant="primary")
                            download_md = gr.Button("Download Markdown", variant="secondary")
                            download_html = gr.Button("Download HTML", variant="secondary")
                            view_in_browser = gr.Button("View in Browser", variant="secondary")
                    
                    with gr.Column(scale=3):
                        with gr.Tabs():
                            with gr.TabItem("Edit Markdown"):
                                edit_md = gr.Code(
                                    language="markdown",
                                    lines=30,
                                    label="Markdown Content"
                                )
                            with gr.TabItem("Preview HTML"):
                                edit_html_preview = gr.HTML()
                        
                        edit_status = gr.Textbox(label="Status", interactive=False)
                
                # File outputs for downloads
                edit_file_output = gr.Files(label="Download Files", visible=False)
        
        # Generation function
        def generate_blog_post_wrapper(topic, focus, date_range, tone, length, temperature, add_toc, seo_optimized):
            if not topic:
                raise gr.Error("Please enter a blog topic")
            
            # Map date range to API values
            date_range_map = {
                "All time": None,
                "Last week": "last_week",
                "Last month": "last_month",
                "Last year": "last_year",
                "Last 1 month": "1m",
                "Last 2 months": "2m",
                "Last 3 months": "3m",
                "Last 4 months": "4m",
                "Last 5 months": "5m",
                "Last 6 months": "6m"
            }
            date_range_val = date_range_map.get(date_range, None)
            
            # Map length to task description
            length_map = {
                "Short (500-1000 words)": "short",
                "Medium (1000-2000 words)": "medium",
                "Long (2000+ words)": "long"
            }
            length_val = length_map.get(length, "medium")
            
            result = generator.generate_blog_post(
                topic=topic,
                focus=focus if focus else "latest trends and developments",
                date_range=date_range_val,
                add_toc=add_toc,
                seo_optimized=seo_optimized,
                tone=tone.lower(),
                length=length_val,
                temperature=temperature
            )
            
            # Prepare files for download
            files = []
            if 'filepath' in result and os.path.exists(result['filepath']):
                files.append(result['filepath'])
            if 'html_path' in result and os.path.exists(result['html_path']):
                files.append(result['html_path'])
            
            # Change the return statement to ensure all outputs are connected:
            return {
                md_output: result.get('content', ''),
                html_output: result.get('html_content', ''),
                file_output: files,
                info_output: {
                    'topic': result.get('topic'),
                    'focus': result.get('focus'),
                    'image_url': result.get('image_url'),
                    'validation_issues': result.get('validation_issues', []),
                    'sentiment': result.get('sentiment'),
                    'file_path': result.get('filepath'),
                    'html_path': result.get('html_path')
                }
            }
        
        # Edit tab functions
        def refresh_post_list():
            return gr.update(choices=generator.list_generated_posts())
        
        def load_post(filepath):
            if not filepath:
                raise gr.Error("Please select a post to load")
            content = generator.load_post_for_editing(filepath)
            html_path = filepath.replace('.md', '.html')
            with open(html_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            return {
                edit_md: content,
                edit_html_preview: html_content,
                edit_status: f"Loaded: {os.path.basename(filepath)}"
            }
        
        def save_post(filepath, content):
            if not filepath:
                raise gr.Error("No post selected for saving")
            result = generator.save_edited_post(filepath, content)
            if result['status'] == 'success':
                files = [filepath, result['html_path']]
                return {
                    edit_html_preview: result['html_content'],
                    edit_status: f"Saved: {os.path.basename(filepath)}",
                    edit_file_output: files
                }
            else:
                return {
                    edit_status: f"Error saving: {result['error']}"
                }
        
        def delete_post(filepath):
            if not filepath:
                raise gr.Error("No post selected for deletion")
            result = generator.delete_post(filepath)
            if result['status'] == 'success':
                return {
                    post_selector: gr.update(choices=generator.list_generated_posts()),
                    edit_md: "",
                    edit_html_preview: "",
                    edit_status: f"Deleted: {os.path.basename(filepath)}"
                }
            else:
                return {
                    edit_status: f"Error deleting: {result['error']}"
                }
        
        def download_markdown(filepath):
            if not filepath:
                raise gr.Error("No post selected")
            return filepath
        
        def download_html_fn(filepath):
            if not filepath:
                raise gr.Error("No post selected")
            return filepath.replace('.md', '.html')
        
        def view_post_in_browser(filepath):
            if not filepath:
                raise gr.Error("No post selected")
            html_path = filepath.replace('.md', '.html')
            webbrowser.open(f"file://{os.path.abspath(html_path)}")
            return {"edit_status": f"Opened in browser: {os.path.basename(html_path)}"}
        
        # Generate tab events
        generate_btn.click(
            fn=generate_blog_post_wrapper,
            inputs=[topic, focus, date_range, tone, length, temperature, add_toc, seo_optimized],
            outputs=[md_output, html_output, file_output, info_output]
        )
        
        # Edit tab events
        refresh_btn.click(
            fn=refresh_post_list,
            outputs=[post_selector]
        )
        
        load_btn.click(
            fn=load_post,
            inputs=[post_selector],
            outputs=[edit_md, edit_html_preview, edit_status]
        )
        
        save_btn.click(
            fn=save_post,
            inputs=[post_selector, edit_md],
            outputs=[edit_html_preview, edit_status, edit_file_output]
        )
        
        delete_btn.click(
            fn=delete_post,
            inputs=[post_selector],
            outputs=[post_selector, edit_md, edit_html_preview, edit_status]
        )
        
        download_md.click(
            fn=download_markdown,
            inputs=[post_selector],
            outputs=[edit_file_output]
        )
        
        download_html.click(
            fn=download_html_fn,
            inputs=[post_selector],
            outputs=[edit_file_output]
        )
        
        view_in_browser.click(
            fn=view_post_in_browser,
            inputs=[post_selector],
            outputs=[edit_status]
        )
    
    return demo
