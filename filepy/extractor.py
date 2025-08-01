# AI Extraction Engine - LangChain + GPT-4 Processing
# requirements.txt additions: langchain, openai, langchain-openai, pydantic, spacy

import asyncio
import json
import re
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import BaseOutputParser
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import spacy

# Load Indonesian NLP model
try:
    nlp = spacy.load("xx_ent_wiki_sm")  # Multilingual model that supports Indonesian
except OSError:
    print("Installing spaCy multilingual model...")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "xx_ent_wiki_sm"])
    nlp = spacy.load("xx_ent_wiki_sm")

class DocumentCategory(Enum):
    CONTRACT = "contract"
    INVOICE = "invoice"
    RECEIPT = "receipt"
    LEGAL_DOCUMENT = "legal_document"
    FINANCIAL_REPORT = "financial_report"
    EMAIL = "email"
    PRESENTATION = "presentation"
    TECHNICAL_DOCUMENT = "technical_document"
    OTHER = "other"

class EntityType(Enum):
    PERSON = "person"
    ORGANIZATION = "organization"
    DATE = "date"
    MONEY = "money"
    LOCATION = "location"
    PHONE = "phone"
    EMAIL = "email"
    ID_NUMBER = "id_number"
    BANK_ACCOUNT = "bank_account"

@dataclass
class ExtractedEntity:
    text: str
    entity_type: EntityType
    confidence: float
    start_pos: int
    end_pos: int
    context: str

class DocumentClassification(BaseModel):
    category: DocumentCategory = Field(description="Primary document category")
    subcategory: str = Field(description="More specific document type")
    confidence: float = Field(description="Classification confidence score")
    key_indicators: List[str] = Field(description="Text patterns that led to this classification")

class DocumentEntities(BaseModel):
    people: List[str] = Field(description="Names of people mentioned")
    organizations: List[str] = Field(description="Company/organization names")
    dates: List[str] = Field(description="Important dates found")
    monetary_amounts: List[str] = Field(description="Money amounts with currency")
    locations: List[str] = Field(description="Addresses and locations")
    contact_info: Dict[str, List[str]] = Field(description="Emails, phones, etc.")
    id_numbers: List[str] = Field(description="ID numbers, account numbers, etc.")

class DocumentSummary(BaseModel):
    executive_summary: str = Field(description="2-3 sentence executive summary")
    key_points: List[str] = Field(description="Main points or findings")
    action_items: List[str] = Field(description="Required actions or next steps")
    urgency_level: str = Field(description="LOW, MEDIUM, HIGH, CRITICAL")
    sentiment: str = Field(description="POSITIVE, NEUTRAL, NEGATIVE")

class TableStructure(BaseModel):
    headers: List[str] = Field(description="Column headers")
    rows: List[List[str]] = Field(description="Table data rows")
    summary: str = Field(description="What this table represents")

class AIExtractionEngine:
    def __init__(self, openai_api_key: str):
        self.llm = ChatOpenAI(
            api_key=openai_api_key,
            model_name="gpt-4-turbo-preview",
            temperature=0.1,
            max_tokens=4000
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        self.indonesian_patterns = {
            'id_number': [
                r'\b\d{16}\b',  # KTP
                r'\b\d{4}-\d{4}-\d{4}-\d{4}\b',  # Formatted ID
                r'NIK\s*:?\s*(\d{16})',  # NIK pattern
            ],
            'phone': [
                r'\+62\s?\d{2,3}\s?\d{3,4}\s?\d{3,4}',  # Indonesian international
                r'08\d{2}\s?\d{3,4}\s?\d{3,4}',  # Local mobile
                r'\(021\)\s?\d{3,4}\s?\d{3,4}',  # Jakarta landline
            ],
            'currency': [
                r'Rp\.?\s?[\d.,]+ juta',
                r'Rp\.?\s?[\d.,]+',
                r'IDR\s?[\d.,]+',
                r'USD\s?\$?[\d.,]+',
            ]
        }
    
    async def extract_all(self, content: str, filename: str) -> Dict[str, Any]:
        """Complete extraction pipeline"""
        
        # Split content for processing
        chunks = self.text_splitter.split_text(content)
        
        # Parallel processing of different extraction tasks
        tasks = [
            self._classify_document(content[:2000], filename),
            self._extract_entities_llm(chunks[0] if chunks else content[:2000]),
            self._summarize_document(content[:3000]),
            self._extract_tables_llm(content),
            self._analyze_sentiment(content[:2000]),
            self._extract_regex_patterns(content)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results
        classification = results[0] if not isinstance(results[0], Exception) else None
        entities = results[1] if not isinstance(results[1], Exception) else None
        summary = results[2] if not isinstance(results[2], Exception) else None
        tables = results[3] if not isinstance(results[3], Exception) else None
        sentiment = results[4] if not isinstance(results[4], Exception) else None
        regex_entities = results[5] if not isinstance(results[5], Exception) else {}
        
        return {
            'classification': classification,
            'entities': entities,
            'summary': summary,
            'tables': tables,
            'sentiment': sentiment,
            'indonesian_entities': regex_entities,
            'processing_metadata': {
                'chunks_processed': len(chunks),
                'extraction_timestamp': datetime.now().isoformat(),
                'content_length': len(content)
            }
        }
    
    async def _classify_document(self, content: str, filename: str) -> DocumentClassification:
        """AI-powered document classification"""
        
        classification_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                """You are an expert document classifier specializing in Indonesian business documents.
                
                Analyze the document content and filename to determine:
                1. Primary category (contract, invoice, legal, financial, etc.)
                2. Specific subcategory 
                3. Confidence level
                4. Key text patterns that indicate this classification
                
                Consider Indonesian business context, language patterns, and document structures.
                
                {format_instructions}"""
            ),
            HumanMessagePromptTemplate.from_template(
                "Filename: {filename}\n\nDocument Content:\n{content}"
            )
        ])
        
        parser = PydanticOutputParser(pydantic_object=DocumentClassification)
        
        formatted_prompt = classification_prompt.format_prompt(
            content=content,
            filename=filename,
            format_instructions=parser.get_format_instructions()
        )
        
        response = await self.llm.ainvoke(formatted_prompt.to_messages())
        return parser.parse(response.content)
    
    async def _extract_entities_llm(self, content: str) -> DocumentEntities:
        """Advanced entity extraction using LLM"""
        
        entity_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                """Extract all relevant entities from this Indonesian business document.
                
                Focus on:
                - People names (Indonesian naming conventions)
                - Company/organization names
                - Important dates (various formats)
                - Monetary amounts (Rupiah, USD, etc.)
                - Locations (addresses, cities)
                - Contact information (emails, phones)
                - ID numbers (KTP, NPWP, etc.)
                
                Be thorough but accurate. Include variations and formatted versions.
                
                {format_instructions}"""
            ),
            HumanMessagePromptTemplate.from_template("Content:\n{content}")
        ])
        
        parser = PydanticOutputParser(pydantic_object=DocumentEntities)
        
        formatted_prompt = entity_prompt.format_prompt(
            content=content,
            format_instructions=parser.get_format_instructions()
        )
        
        response = await self.llm.ainvoke(formatted_prompt.to_messages())
        return parser.parse(response.content)
    
    async def _summarize_document(self, content: str) -> DocumentSummary:
        """Generate executive summary and key insights"""
        
        summary_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                """Create a comprehensive business summary of this document.
                
                Provide:
                1. Executive summary (2-3 sentences)
                2. Key points/findings
                3. Action items or next steps
                4. Urgency assessment
                5. Overall sentiment
                
                Focus on business implications and actionable insights.
                Use Indonesian business context when relevant.
                
                {format_instructions}"""
            ),
            HumanMessagePromptTemplate.from_template("Document:\n{content}")
        ])
        
        parser = PydanticOutputParser(pydantic_object=DocumentSummary)
        
        formatted_prompt = summary_prompt.format_prompt(
            content=content,
            format_instructions=parser.get_format_instructions()
        )
        
        response = await self.llm.ainvoke(formatted_prompt.to_messages())
        return parser.parse(response.content)
    
    async def _extract_tables_llm(self, content: str) -> List[TableStructure]:
        """Extract and structure table data using LLM"""
        
        # First, identify if tables exist
        table_detection_prompt = f"""
        Analyze this content and identify any tabular data structures.
        Look for:
        - Formatted tables with headers and rows
        - List structures that could be tables
        - Financial data in columns
        - Structured information layouts
        
        Content preview: {content[:1000]}...
        
        Response format: JSON array of table descriptions or empty array if no tables.
        """
        
        detection_response = await self.llm.ainvoke(table_detection_prompt)
        
        if "no tables" in detection_response.content.lower() or "[]" in detection_response.content:
            return []
        
        # Extract detailed table structures
        table_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                """Extract all table structures from this document.
                
                For each table:
                1. Identify column headers
                2. Extract all data rows
                3. Provide a summary of what the table represents
                
                Maintain data accuracy and structure.
                
                {format_instructions}"""
            ),
            HumanMessagePromptTemplate.from_template("Content:\n{content}")
        ])
        
        # Custom parser for multiple tables
        table_response = await self.llm.ainvoke(f"Extract tables from:\n{content[:2000]}")
        
        # Parse response and structure as TableStructure objects
        # This is a simplified version - you'd want more robust parsing
        try:
            tables_data = json.loads(table_response.content)
            return [TableStructure(**table) for table in tables_data if isinstance(table, dict)]
        except:
            return []
    
    async def _analyze_sentiment(self, content: str) -> Dict[str, str]:
        """Analyze document sentiment and tone"""
        
        sentiment_prompt = f"""
        Analyze the sentiment and tone of this business document:
        
        {content[:1500]}
        
        Provide:
        1. Overall sentiment: POSITIVE, NEUTRAL, NEGATIVE
        2. Tone: FORMAL, INFORMAL, URGENT, FRIENDLY, etc.
        3. Confidence level: HIGH, MEDIUM, LOW
        4. Key phrases that indicate this sentiment
        
        Response as JSON.
        """
        
        response = await self.llm.ainvoke(sentiment_prompt)
        
        try:
            return json.loads(response.content)
        except:
            return {
                "sentiment": "NEUTRAL",
                "tone": "FORMAL",
                "confidence": "MEDIUM",
                "key_phrases": []
            }
    
    async def _extract_regex_patterns(self, content: str) -> Dict[str, List[str]]:
        """Extract Indonesian-specific patterns using regex"""
        
        results = {}
        
        for pattern_type, patterns in self.indonesian_patterns.items():
            matches = []
            for pattern in patterns:
                found = re.findall(pattern, content, re.IGNORECASE)
                matches.extend(found)
            
            # Remove duplicates and clean
            results[pattern_type] = list(set(matches))
        
        # SpaCy NER for additional entities
        doc = nlp(content[:1000])  # Limit for performance
        
        spacy_entities = {
            'persons': [ent.text for ent in doc.ents if ent.label_ == "PERSON"],
            'organizations': [ent.text for ent in doc.ents if ent.label_ == "ORG"],
            'locations': [ent.text for ent in doc.ents if ent.label_ in ["GPE", "LOC"]],
            'dates': [ent.text for ent in doc.ents if ent.label_ == "DATE"]
        }
        
        results.update(spacy_entities)
        return results

# FastAPI Integration
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

class ExtractionRequest(BaseModel):
    content: str
    filename: str = "document.txt"

class ExtractionResponse(BaseModel):
    status: str
    data: Dict[str, Any]
    processing_time: float

# Initialize extraction engine
extraction_engine = None

def init_extraction_engine(api_key: str):
    global extraction_engine
    extraction_engine = AIExtractionEngine(api_key)

@app.post("/extract-entities", response_model=ExtractionResponse)
async def extract_entities_endpoint(request: ExtractionRequest):
    """Advanced AI extraction endpoint"""
    if not extraction_engine:
        raise HTTPException(status_code=500, detail="Extraction engine not initialized")
    
    start_time = asyncio.get_event_loop().time()
    
    try:
        results = await extraction_engine.extract_all(request.content, request.filename)
        processing_time = asyncio.get_event_loop().time() - start_time
        
        return ExtractionResponse(
            status="success",
            data=results,
            processing_time=round(processing_time, 2)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")

# Usage example
if __name__ == "__main__":
    # Initialize with your OpenAI API key
    import os
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        init_extraction_engine(api_key)
        print("AI Extraction Engine initialized successfully!")
    else:
        print("Please set OPENAI_API_KEY environment variable")
