from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import ChatPromptTemplate
from llama_index.core import Settings
from dotenv import load_dotenv
import os
import openai


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

Settings.llm = Ollama(model="gemma2", request_timeout=1000);

categories = ["Makeup", "Skin Care", "Hair Care", "Fragrance", "Tools & Accessories", "Shave & hair removal", "Personal Care", "Salon & Spa Equipment", "Foot, Hand & Nail Care"]

subcategories = [["Body", "Eyes", "Face", "Lips", "Makeup Palettes", "Makeup Remover", "Makeup Sets" ], ["Body", "Eyes", "Face", "Lip Care", "Maternity", "Sets & Kits", "Sunscreens & Tanning Products"], ["Detanglers", "Hair Accessories",	 "Hair Coloring Products",	"Hair Cutting Tools",	"Hair Extensions, Wigs & Accessories",	"Hair Fragrances",	"Hair Loss Products",	"Hair Masks",	"Hair Perms, Relaxers & Texturizers",	"Hair Treatment Oils",	"Scalp Treatments",	"Shampoo & Conditioner"	,"Styling Products"], ["Children's", "Dusting Powders", "Men's", "Sets", "Women's"], ["Bags & Cases",		"Bathing Accessories",	"Cotton Balls & Swabs",	"Makeup Brushes & Tools",	"Mirrors",	"Refillable Containers",	"Shave & Hair Removal"	], ["Men's", "Women's"], ["	Bath & Bathing Accessories", "Deodorants & Antiperspirants", "Lip Care", "Oral Care", "Piercing & Tattoo Supplies", "Scrubs & Body Treatments", "Shave & Hair Removal"], ["Hair Drying Hoods"	,"Galvanic Facial Machine","Handheld Mirrors"	,"High-Frequency Facial Machines"	,"Manicure Tables"	,"Professional Massage Equipment","Salon & Spa Stools","Spa Beds & Tables","Salon & Spa Chairs","Spa Storage Systems","Spa Hot Towel Warmers"]]

question = "Recommend me 2-3 artificial eyebrows"

context = f"Given {categories} and {subcategories} where each index of {subcategories} represent subcategories of array categories:{categories}"

template = f"""You'll be given a list a categories with their respective subcategories in the given context
            ==========================================================================================
            Context: {context}
            ==========================================================================================

            Based on your prior knowledge, you'll have to tell under which category and subcategory does the product given in the below query fall under
            Respond only with "category", "subcategory" where <category> is the category of product in the query and <subcategory> is the subcateogry of product in the query.

            =========================================================================================
            Question: {question}
            =========================================================================================
            """

print(Settings.llm.complete(template))