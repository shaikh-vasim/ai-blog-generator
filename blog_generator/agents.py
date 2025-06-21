"""
AI-Powered Blog Post Generator
This module implements a comprehensive blog post generation system using multiple AI agents
to create high-quality, well-researched technical content. The system leverages CrewAI
framework with Google's Gemini model and external APIs for web search and image retrieval.
Classes:
BlogPostGenerator: Main class that orchestrates the content generation pipeline
Dependencies:
- crewai: Framework for multi-agent AI systems
- langchain_google_genai: Google Gemini AI integration
- requests: HTTP requests for API calls
- nltk: Natural language processing for sentiment analysis
- markdown: Markdown to HTML conversion
- gradio: Web interface for user interaction
Environment Variables Required:
- GOOGLE_API_KEY: Google Gemini API key
- SERPER_API_KEY: Serper web search API key (optional)
- UNSPLASH_ACCESS_KEY: Unsplash image API key (optional)
Example Usage:
>>> generator = BlogPostGenerator()
>>> result = generator.generate_blog_post(
...     topic="Artificial Intelligence",
...     focus="practical applications",
...     date_range="last_month",
...     add_toc=True,
...     seo_optimized=True,
...     tone="professional",
...     length="comprehensive",
...     temperature=0.7
... )
>>> print(f"Generated post: {result['filepath']}")
"""


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

# Initialize Gemini with proper error handling
try:
    from crewai import LLM

    llm = LLM(
        model=os.getenv('GOOGLE_MODEL_NAME'),
        temperature=0.7,
        api_key=os.getenv('GOOGLE_API_KEY')
    )
    logger.info("*****************************************")
    logger.info("Successfully initialized Gemini model")
except Exception as e:
    logger.error(f"Failed to initialize Gemini: {str(e)}")
    raise

class BlogPostGenerator:
    def __init__(self):
        self.output_folder = "generated_posts"
        os.makedirs(self.output_folder, exist_ok=True)
        
        # Initialize NLTK components
        try:
            nltk.download('vader_lexicon', quiet=True)
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
            logger.info("Successfully initialized sentiment analyzer")
        except Exception as e:
            logger.warning(f"Failed to initialize NLTK components: {str(e)}")
            self.sentiment_analyzer = None
        
        # API keys validation
        self.serper_api_key = os.getenv('SERPER_API_KEY')
        self.unsplash_api_key = os.getenv('UNSPLASH_ACCESS_KEY')
        
        if not self.serper_api_key:
            logger.warning("SERPER_API_KEY not found in environment variables")
        
        if not self.unsplash_api_key:
            logger.warning("UNSPLASH_ACCESS_KEY not found in environment variables")
        
        # Default tech-related images
        self.default_images = [
            "https://images.unsplash.com/photo-1620712943543-bcc4688e7485",
            "https://images.unsplash.com/photo-1550751827-4bd374c3f58b",
            "https://images.unsplash.com/photo-1526374965328-7f61d4dc18c5",
            "https://images.unsplash.com/photo-1518770660439-4636190af475",
            "https://images.unsplash.com/photo-1461749280684-dccba630e2f6",
            "https://images.unsplash.com/photo-1558346547-4439467bd1d5",
            "https://images.unsplash.com/photo-1535378620166-273708d44e4c",
            "https://images.unsplash.com/photo-1531297484001-80022131f5a1",
            "https://images.unsplash.com/photo-1555949963-ff9fe0c870eb",
            "https://images.unsplash.com/photo-1581472723648-909f4851d4ae"
        ]
        
        # Initialize agents
        self.initialize_agents()
    
    def initialize_agents(self):
        """Initialize all agents with their roles and tools"""
        # Create the search_web and find_image tools bound to this instance
        search_web_tool = tool("WebSearch")(self.search_web)
        find_image_tool = tool("ImageFinder")(self.find_image)
        
        # Researcher agent
        self.researcher = Agent(
            role='Senior Research Analyst',
            goal='Find and analyze the latest and most relevant information on given topics',
            backstory="""You're an expert researcher specialized in technology, AI, and GenAI.
            You have a knack for finding cutting-edge information and identifying key trends.
            You're thorough and make sure to verify your sources. You always include specific dates
            and citations for all facts and figures.""",
            verbose=True,
            # allow_delegation=True,
            llm=llm,
            tools=[search_web_tool]
        )
        
        # Writer agent
        self.writer = Agent(
            role='Tech Content Writer',
            goal='Write engaging, detailed blog posts about technology topics',
            backstory="""You're an experienced technical writer with expertise in AI and GenAI.
            You can explain complex concepts in simple terms and create well-structured content.
            You include relevant examples and code snippets when appropriate. Your writing is 
            concise, engaging, and always includes a strong hook and conclusion.""",
            verbose=True,
            # allow_delegation=True,
            llm=llm
        )
        
        # Editor agent
        self.editor = Agent(
            role='Senior Editor',
            goal='Ensure the content is high quality, accurate, and well-structured',
            backstory="""You're a meticulous editor with deep knowledge of technology content.
            You improve clarity, fix errors, and ensure the content meets high standards.
            You also verify technical accuracy and suggest improvements. You're particularly
            vigilant about maintaining proper transitions between paragraphs and sections.""",
            verbose=True,
            # allow_delegation=True,
            llm=llm
        )
        
        # Illustrator agent
        self.illustrator = Agent(
            role='Content Illustrator',
            goal='Find or create appropriate visual elements for the blog post',
            backstory="""You specialize in finding or describing appropriate visuals for technical content.
            You understand what makes a good technical illustration and can find free-to-use images.
            You also suggest diagrams or charts when needed. You always provide detailed alt text for
            accessibility and ensure images enhance rather than distract from the content.""",
            verbose=True,
            # allow_delegation=True,
            llm=llm,
            tools=[find_image_tool]
        )
        
        # Fact checker agent
        self.fact_checker = Agent(
            role='Technical Fact Checker',
            goal='Verify all technical claims and ensure information accuracy',
            backstory="""You're a thorough fact checker with expertise in technology and AI.
            You verify all technical claims, dates, statistics, and references. You flag
            potential inaccuracies and ensure the content is up-to-date and trustworthy.
            You also check for technical inconsistencies and logical errors.""",
            verbose=True,
            # allow_delegation=True,
            llm=llm,
            tools=[search_web_tool]
        )
    
    def _generate_cache_key(self, func_name: str, args: tuple, kwargs: Dict[str, Any]) -> str:
        """Generate a unique cache key based on function name and arguments"""
        args_str = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(f"{func_name}:{args_str}".encode()).hexdigest()
    
    def search_web(self, query: str, date_range: Optional[str] = None) -> str:
        """Search the web for current information on topics."""
        if not self.serper_api_key:
            logger.warning("No Serper API key available, returning mock data")
            return f"""Mock search results for '{query}':
            
1. Title: {query} - An Overview
Snippet: This is a mock result since no API key was provided. In a real implementation, this would show actual search results.
URL: https://example.com

2. Title: Recent Developments in {query}
Snippet: Mock data showing how search results would appear with proper API configuration.
URL: https://example.com
"""
        
        params = {
            'q': query,
            'gl': 'us',
            'hl': 'en',
            'num': 5
        }
        
        if date_range:
            if date_range == 'last_week':
                date_filter = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
                params['timePeriod'] = f'custom:{date_filter}:{datetime.now().strftime("%Y-%m-%d")}'
            elif date_range == 'last_month':
                params['timePeriod'] = 'm1'
            elif date_range == 'last_year':
                params['timePeriod'] = 'y1'
            elif date_range in ['1m', '2m', '3m', '4m', '5m', '6m']:
                months = int(date_range.replace('m', ''))
                start_date = (datetime.now() - timedelta(days=30 * months)).strftime('%Y-%m-%d')
                end_date = datetime.now().strftime('%Y-%m-%d')
                params['timePeriod'] = f'custom:{start_date}:{end_date}'

        headers = {
            'X-API-KEY': self.serper_api_key,
            'Content-Type': 'application/json'
        }
        
        try:
            response = requests.post(
                'https://google.serper.dev/search',
                headers=headers,
                data=json.dumps(params),
                timeout=10
            )
            response.raise_for_status()
            results = response.json()
            
            formatted_results = []
            
            if 'organic' in results:
                for item in results['organic'][:5]:
                    formatted_result = f"Title: {item.get('title', 'No title')}\n"
                    formatted_result += f"Snippet: {item.get('snippet', 'No snippet')}\n"
                    formatted_result += f"URL: {item.get('link', '#')}\n\n"
                    formatted_results.append(formatted_result)
            
            if not formatted_results:
                return f"No relevant search results were found for '{query}'."
            
            return "\n".join(formatted_results)
        except requests.exceptions.RequestException as e:
            logger.error(f"Search API request failed: {str(e)}")
            return f"The search for '{query}' failed due to a network or API error."
    
    def find_image(self, query: str) -> str:
        """Find appropriate images for blog posts based on a query."""
        if self.unsplash_api_key:
            try:
                url = f"https://api.unsplash.com/search/photos"
                headers = {"Authorization": f"Client-ID {self.unsplash_api_key}"}
                params = {"query": query, "per_page": 1}
                
                response = requests.get(url, headers=headers, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                if data["results"]:
                    image_url = data["results"][0]["urls"]["regular"]
                    logger.info(f"Found image from Unsplash for query: {query}")
                    return image_url
            except Exception as e:
                logger.warning(f"Unsplash API request failed: {str(e)}")
        
        logger.info(f"Using default image for query: {query}")
        return random.choice(self.default_images)
    
    def _analyze_sentiment(self, text: str) -> Optional[dict]:
        """Analyze the sentiment of text to ensure positive/neutral tone"""
        if not self.sentiment_analyzer:
            return None
        
        try:
            sentiment = self.sentiment_analyzer.polarity_scores(text)
            return sentiment
        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {str(e)}")
            return None
    
    def _clean_markdown(self, content: str) -> str:
        """Clean up markdown content by removing unwanted code block markers"""
        # Remove ```markdown markers if they exist
        content = content.replace("```markdown", "").replace("```", "")
        
        # Remove any remaining triple backticks that might be alone
        content = re.sub(r'^```\s*$', '', content, flags=re.MULTILINE)
        
        # Remove any language specifiers from code blocks
        content = re.sub(r'```[a-zA-Z]+\n', '```\n', content)
        
        return content.strip()
    
    def _validate_content(self, content: str) -> list:
        """Validate content for common issues"""
        issues = []
        
        if len(content) < 1000:
            issues.append("Content may be too short for a comprehensive blog post")
        
        if "![" not in content and "](http" not in content:
            issues.append("No images found in content")
        
        if "# " not in content and "## " not in content:
            issues.append("No proper header structure found")
        
        if any(tech in content.lower() for tech in ["code", "programming", "python", "javascript"]):
            if "```" not in content:
                issues.append("Technical topic with no code examples")
        
        if not re.search(r'\[\d+\]|\[\w+\d*\]|Source:', content):
            issues.append("No citations or references found")
        
        return issues
    
    def _create_tasks(self, topic: str, focus: str, tone: str, length: str):
        """Create tasks for the crew with tone and length parameters"""
        # Research task
        research_task = Task(
            description=f"""Conduct thorough research on '{topic}' with focus on {focus}.
            Search the web for the most relevant and recent information.""",
            expected_output=f"""A comprehensive research report on '{topic}' covering:
            1. Topic overview
            2. Key concepts
            3. Current trends
            4. Applications
            5. Statistics with sources
            6. References with links
            7. Controversies
            8. Future development""",
            agent=self.researcher
        )
        
        # Writing task with tone and length parameters
        write_task = Task(
            description=f"""Write a {length} blog post about '{topic}' focusing on {focus} in a {tone} tone.
            The post should be informative yet accessible, suitable for a technical audience.
            DO NOT include markdown code block markers (```) in the output.""",
            expected_output=f"""A complete blog post in markdown format about '{topic}' with:
            1. SEO-optimized title
            2. Engaging introduction
            3. Clear section headers
            4. Detailed explanations
            5. Code snippets (if applicable)
            6. Cited sources
            7. Actionable insights
            8. Strong conclusion""",
            agent=self.writer,
            context=[research_task]
        )
        
        # Fact checking task
        fact_check_task = Task(
            description=f"""Review the blog post about '{topic}' and verify all technical claims.""",
            expected_output="A detailed fact-checking report with specific corrections",
            agent=self.fact_checker,
            context=[write_task]
        )
        
        # Editing task
        edit_task = Task(
            description=f"""Review and edit the blog post about '{topic}'.
            Ensure technical accuracy, clear writing style, and proper structure.
            Remove any unnecessary markdown code block markers.""",
            expected_output="A polished, publication-ready blog post in markdown format",
            agent=self.editor,
            context=[write_task, fact_check_task]
        )
        
        # Illustration task
        illustrate_task = Task(
            description=f"""Find appropriate visual elements for the blog post about '{topic}'.
            Include featured image and supporting images at key points.""",
            expected_output="Markdown formatted image references positioned in the post",
            agent=self.illustrator,
            context=[edit_task]
        )
        
        return [research_task, write_task, fact_check_task, edit_task, illustrate_task]
    
    def generate_blog_post(self, topic: str, focus: str, date_range: str, 
                         add_toc: bool, seo_optimized: bool, tone: str, 
                         length: str, temperature: float) -> dict:
        """Generate a single blog post with given parameters"""
        logger.info(f"Generating post for topic: {topic}")
        
        # Update LLM temperature
        llm.temperature = temperature
        
        # Create tasks for the crew
        tasks = self._create_tasks(topic, focus, tone, length)
        
        # Create crew and execute tasks
        crew = Crew(
            agents=[self.researcher, self.writer, self.fact_checker, self.editor, self.illustrator],
            tasks=tasks,
            verbose=True,
            process=Process.sequential
        )
        
        try:
            result = crew.kickoff()
            
            # Extract the actual content string from CrewOutput object
            if hasattr(result, 'raw_output'):
                content = result.raw_output
            elif hasattr(result, 'result'):
                content = result.result
            elif hasattr(result, 'output'):
                content = result.output
            else:
                content = str(result)
            
            # Clean the markdown content
            content = self._clean_markdown(content)
            
            validation_issues = self._validate_content(content)
            if validation_issues:
                logger.warning(f"Content validation issues: {validation_issues}")
            
            image_url = self.find_image(topic)
            sentiment = self._analyze_sentiment(content[:1000]) if len(content) > 0 else None
            
            clean_topic = re.sub(r'[^\w\s-]', '', topic).strip().lower()
            clean_topic = re.sub(r'[-\s]+', '-', clean_topic)
            filename = f"{clean_topic}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            filepath = os.path.join(self.output_folder, filename)
            
            final_content = content
            if "![Featured Image]" not in final_content:
                header = f"# {topic}\n\n"
                reading_time = max(5, len(final_content) // 1000)
                header += f"> **Reading time:** {reading_time} min | **Difficulty:** Intermediate | **Published:** {datetime.now().strftime('%B %d, %Y')}\n\n"
                header += f"![Featured Image]({image_url})\n\n"
                final_content = header + final_content
            
            # Add table of contents if requested
            if add_toc and "## Table of Contents" not in final_content:
                headings = re.findall(r'^(#{2,4})\s+(.+)$', final_content, re.MULTILINE)
                if headings:
                    toc = "## Table of Contents\n\n"
                    for level, title in headings:
                        indent = "  " * (len(level) - 2)
                        slug = re.sub(r'[^\w\s-]', '', title).strip().lower()
                        slug = re.sub(r'[-\s]+', '-', slug)
                        toc += f"{indent}- [{title}](#{slug})\n"
                    toc += "\n"
                    
                    intro_end = re.search(r'^#{2}\s+.+$', final_content, re.MULTILINE)
                    if intro_end:
                        pos = intro_end.end()
                        final_content = final_content[:pos] + "\n\n" + toc + final_content[pos:]
                    else:
                        first_heading = re.search(r'^#\s+.+$', final_content, re.MULTILINE)
                        if first_heading:
                            pos = first_heading.end()
                            final_content = final_content[:pos] + "\n\n" + toc + final_content[pos:]
            
            # Write the markdown file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(final_content)
            
            # Generate HTML preview with better styling
            html_content = markdown.markdown(final_content, extensions=['extra', 'codehilite', 'tables'])
            html_path = filepath.replace('.md', '.html')
            
            # Enhanced HTML template with better styling
            html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{topic}</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/styles/github.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/highlight.min.js"></script>
    <script>hljs.highlightAll();</script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.8;
            color: #333;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f9f9f9;
        }}
        h1, h2, h3, h4 {{
            color: #2c3e50;
            margin-top: 1.5em;
            margin-bottom: 0.5em;
        }}
        h1 {{
            border-bottom: 2px solid #eee;
            padding-bottom: 10px;
        }}
        img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin: 20px 0;
        }}
        code {{
            font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
            background-color: #f5f5f5;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.9em;
        }}
        pre {{
            background-color: #f5f5f5;
            padding: 16px;
            border-radius: 8px;
            overflow-x: auto;
            margin: 20px 0;
            border-left: 4px solid #3498db;
        }}
        blockquote {{
            border-left: 4px solid #3498db;
            padding-left: 16px;
            margin-left: 0;
            color: #555;
            font-style: italic;
        }}
        a {{
            color: #3498db;
            text-decoration: none;
        }}
        a:hover {{
            text-decoration: underline;
        }}
        .toc {{
            background-color: #f0f7ff;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }}
        .toc ul {{
            padding-left: 20px;
        }}
        .toc li {{
            margin-bottom: 8px;
        }}
        .meta {{
            color: #7f8c8d;
            font-size: 0.9em;
            margin-bottom: 20px;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        th {{
            background-color: #f2f2f2;
        }}
    </style>
</head>
<body>
    {html_content}
</body>
</html>"""
            
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_template)
            
            logger.info(f"Successfully generated blog post: {filepath}")
            
            return {
                'topic': topic,
                'focus': focus,
                'content': final_content,
                'html_content': html_template,
                'image_url': image_url,
                'filepath': filepath,
                'html_path': html_path,
                'validation_issues': validation_issues,
                'sentiment': sentiment
            }
            
        except Exception as e:
            logger.error(f"Error generating post for '{topic}': {str(e)}")
            return {
                'error': str(e),
                'topic': topic,
                'content': f"Error generating content: {str(e)}"
            }
    
    def list_generated_posts(self) -> list:
        """List all generated blog posts"""
        files = []
        for file in os.listdir(self.output_folder):
            if file.endswith('.md'):
                files.append(os.path.join(self.output_folder, file))
        return sorted(files, key=os.path.getmtime, reverse=True)
    
    def load_post_for_editing(self, filepath: str) -> str:
        """Load a post for editing"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error loading post {filepath}: {str(e)}")
            return f"Error loading post: {str(e)}"
    
    def save_edited_post(self, filepath: str, content: str) -> dict:
        """Save an edited post and regenerate HTML"""
        try:
            # Save the markdown file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Regenerate HTML
            html_path = filepath.replace('.md', '.html')
            html_content = markdown.markdown(content, extensions=['extra', 'codehilite', 'tables'])
            
            # Use the same enhanced HTML template
            html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Edited Post</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/styles/github.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/highlight.min.js"></script>
    <script>hljs.highlightAll();</script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.8;
            color: #333;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f9f9f9;
        }}
        h1, h2, h3, h4 {{
            color: #2c3e50;
            margin-top: 1.5em;
            margin-bottom: 0.5em;
        }}
        h1 {{
            border-bottom: 2px solid #eee;
            padding-bottom: 10px;
        }}
        img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin: 20px 0;
        }}
        code {{
            font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
            background-color: #f5f5f5;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.9em;
        }}
        pre {{
            background-color: #f5f5f5;
            padding: 16px;
            border-radius: 8px;
            overflow-x: auto;
            margin: 20px 0;
            border-left: 4px solid #3498db;
        }}
        blockquote {{
            border-left: 4px solid #3498db;
            padding-left: 16px;
            margin-left: 0;
            color: #555;
            font-style: italic;
        }}
        a {{
            color: #3498db;
            text-decoration: none;
        }}
        a:hover {{
            text-decoration: underline;
        }}
        .toc {{
            background-color: #f0f7ff;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }}
        .toc ul {{
            padding-left: 20px;
        }}
        .toc li {{
            margin-bottom: 8px;
        }}
        .meta {{
            color: #7f8c8d;
            font-size: 0.9em;
            margin-bottom: 20px;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        th {{
            background-color: #f2f2f2;
        }}
    </style>
</head>
<body>
    {html_content}
</body>
</html>"""
            
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_template)
            
            return {
                'status': 'success',
                'filepath': filepath,
                'html_path': html_path,
                'html_content': html_template
            }
        except Exception as e:
            logger.error(f"Error saving edited post {filepath}: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def delete_post(self, filepath: str) -> dict:
        """Delete a generated post and its HTML version"""
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
            
            html_path = filepath.replace('.md', '.html')
            if os.path.exists(html_path):
                os.remove(html_path)
            
            return {'status': 'success'}
        except Exception as e:
            logger.error(f"Error deleting post {filepath}: {str(e)}")
            return {'status': 'error', 'error': str(e)}
