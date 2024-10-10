from flask import Flask, request
from linebot import LineBotApi
from linebot.models import FlexSendMessage, TextSendMessage, QuickReply, QuickReplyButton, MessageAction
from sentence_transformers import SentenceTransformer, util
from neo4j import GraphDatabase  # สำหรับเชื่อมต่อกับ Neo4j
import json
import numpy as np
from selenium import webdriver
import chromedriver_autoinstaller
from bs4 import BeautifulSoup
from selenium.webdriver.chrome.options import Options
import urllib.parse
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Setup Chrome options for Selenium
chrome_options = Options()
chrome_options.add_argument('--headless')  # Run in headless mode
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')
chrome_options.add_argument('--disable-gpu')
chromedriver_autoinstaller.install()

# Initialize SentenceTransformer model
model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')

# Neo4j connection setup
NEO4J_URI = "bolt://localhost:7687"  # Your Neo4j URI
NEO4J_USER = "neo4j"  # Your Neo4j username
NEO4J_PASSWORD = "password"  # Your Neo4j password

# Initialize Neo4j driver
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

app = Flask(__name__)

# Initialize LineBotApi with your channel access token
line_bot_api = LineBotApi('access_token')

# Function to run Neo4j queries
def run_query(query, parameters=None):
    with driver.session() as session:
        result = session.run(query, parameters)
        return [record for record in result]

# Fetching the greeting corpus from Neo4j
cypher_query = '''
MATCH (n:Greeting) RETURN n.name as name, n.msg_reply as reply;
'''
greeting_corpus = []
results = run_query(cypher_query)
for record in results:
    greeting_corpus.append(record['name'])
greeting_corpus = list(set(greeting_corpus))

# Function to compute similarity between user input and greetings
def compute_similar(sentence):
    greeting_vec = model.encode(greeting_corpus, convert_to_tensor=True, normalize_embeddings=True)
    user_vec = model.encode(sentence, convert_to_tensor=True, normalize_embeddings=True)
    similarities = util.cos_sim(greeting_vec, user_vec)
    greeting_np = similarities.cpu().numpy()
    
    # Find the most similar greeting
    max_greeting_score = np.argmax(greeting_np)
    
    if greeting_np[max_greeting_score] > 0.5:  # Threshold for greeting match
        match_greeting = greeting_corpus[max_greeting_score]
        return match_greeting
    return None

# Function to fetch a response message from Neo4j based on the matched greeting
def neo4j_search(greeting):
    query = f"MATCH (n:Greeting) WHERE n.name = '{greeting}' RETURN n.msg_reply as reply"
    results = run_query(query)
    for record in results:
        return record['reply']
    return "Hello! How can I assist you?"

# Function to create the main Quick Reply options (Shoe, Collection, Product, Sale!!, General)
def main_quick_reply():
    return QuickReply(items=[
        QuickReplyButton(action=MessageAction(label="Shoe", text="Shoe")),
        QuickReplyButton(action=MessageAction(label="Collection", text="Collection")),
        QuickReplyButton(action=MessageAction(label="Product", text="Product")),
        QuickReplyButton(action=MessageAction(label="Sale!!", text="Sale!!")),
        QuickReplyButton(action=MessageAction(label="General", text="General"))  # Added General option
    ])


# Function to perform web scraping using Selenium and BeautifulSoup
def scrape_converse(search_term):
    base_url = "https://mustardsneakers.com"
    url = f"{base_url}/search?type=product%2Carticle%2Cpage%2Ccollection&options%5Bprefix%5D=last&q={search_term}"
    driver = webdriver.Chrome(options=chrome_options)
    driver.get(url)
    driver.implicitly_wait(5)  # Wait for the page to load

    # Parse the page source with BeautifulSoup
    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")
    
    product_elements = soup.find_all("div", {"class": "grid-product__content"})
    products_details = []
    
    for product in product_elements:
        title = product.find("div", class_="grid-product__title--body")
        price = product.find("div", class_="grid-product__price")
        image = product.find("img", class_="lazyloaded")
        link = product.find("a", href=True)

        if title and price and image and link:
            # Handle relative URL for images and product links
            img_url = image.get('src') or image.get('data-srcset').split(',')[0].split(' ')[0]
            img_url = urllib.parse.urljoin(base_url, img_url)  # Convert to absolute URL if it's relative
            product_url = urllib.parse.urljoin(base_url, link['href'])  # Get product link

            products_details.append({
                'name': title.text.strip(),
                'price': price.text.strip(),
                'image': img_url,
                'url': product_url  # Add product URL to the details
            })
    
    driver.quit()
    return products_details

# ฟังก์ชันสแครปข้อมูลจากหน้า Best Selling
def scrape_best_selling():
    base_url = "https://mustardsneakers.com"
    url = f"{base_url}/collections/all?sort_by=best-selling"
    
    driver = webdriver.Chrome(options=chrome_options)
    driver.get(url)
    driver.implicitly_wait(5)  # รอให้หน้าโหลดเสร็จ

    # Parse the page source with BeautifulSoup
    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")
    
    # ดึงข้อมูลสินค้าจากหน้า Best Selling
    product_elements = soup.find_all("div", class_="grid-product__content", limit=8)  # ดึงมาแค่ 8 ชิ้นแรก
    products_details = []
    
    for product in product_elements:
        title = product.find("div", class_="grid-product__title--body")
        price = product.find("div", class_="grid-product__price")
        image = product.find("img", class_="lazyloaded")
        link = product.find("a", href=True)

        if title and price and image and link:
            # Handle relative URL for images and product links
            img_url = image.get('src') or image.get('data-srcset').split(',')[0].split(' ')[0]
            img_url = urllib.parse.urljoin(base_url, img_url)  # Convert to absolute URL if it's relative
            product_url = urllib.parse.urljoin(base_url, link['href'])  # Get product link

            products_details.append({
                'name': title.text.strip(),
                'price': price.text.strip(),
                'image': img_url,
                'url': product_url  # Add product URL to the details
            })

    driver.quit()
    return products_details

# Function to scrape specific FAQ answers from the Mustard Sneakers website

def scrape_general_faq():
    base_url = "https://mustardsneakers.com"
    url = f"{base_url}/pages/faq"
    
    driver = webdriver.Chrome(options=chrome_options)
    driver.get(url)
    driver.implicitly_wait(5)  # Wait for the page to load

    # Parse the page source with BeautifulSoup
    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")

    # Extract the relevant <p> elements based on their corresponding questions
    faqs = {}

    # Get all FAQ content areas using the shared class
    faq_contents = soup.find_all("div", {"class": "collapsible-content__inner collapsible-content__inner--faq rte"})

    # Check if we have enough FAQ items
    if len(faq_contents) >= 4:
        # Map each content to the corresponding question
        faqs['1'] = faq_contents[6].find("p").text.strip()  # ลองสินค้าจริงได้ที่ไหนบ้าง
        faqs['2'] = faq_contents[7].find("p").text.strip()  # Mustard Sneakers เป็นแบรนด์ของที่ไหน
        faqs['3'] = faq_contents[8].find("p").text.strip()  # รองเท้าทำมาจากวัสดุอะไร
        faqs['4'] = faq_contents[9].find("p").text.strip()  # ทำความสะอาดรองเท้าอย่างไร

    driver.quit()
    return faqs

def general_quick_reply():
    return QuickReply(items=[
        QuickReplyButton(action=MessageAction(label="1", text="1")),  # FAQ 1
        QuickReplyButton(action=MessageAction(label="2", text="2")),  # FAQ 2
        QuickReplyButton(action=MessageAction(label="3", text="3")),  # FAQ 3
        QuickReplyButton(action=MessageAction(label="4", text="4")),  # FAQ 4
        QuickReplyButton(action=MessageAction(label="Back", text="menu"))  # Go back to main menu
    ])

# ฟังก์ชันเพื่อส่ง Flex Message สำหรับสินค้า Best Selling
def send_best_selling_flex_message(reply_token):
    products = scrape_best_selling()
    
    if not products:
        text_message = TextSendMessage(text="ไม่พบสินค้าที่แนะนำ.")
        line_bot_api.reply_message(reply_token, text_message)
        return

    # Generate Flex Message bubbles with button linking to the product URL
    bubbles = [{
        "type": "bubble",
        "hero": {
            "type": "image",
            "url": prod['image'],
            "size": "full",
            "aspectRatio": "20:13",
            "aspectMode": "cover"
        },
        "body": {
            "type": "box",
            "layout": "vertical",
            "contents": [
                {"type": "text", "text": prod['name'], "weight": "bold", "size": "md", "wrap": True},
                {"type": "text", "text": f"Price: {prod['price']}", "size": "sm", "color": "#999999"}
            ]
        },
        "footer": {
            "type": "box",
            "layout": "vertical",
            "spacing": "sm",
            "contents": [
                {
                    "type": "button",
                    "style": "primary",  # ใช้สีเพื่อให้ปุ่มโดดเด่น
                    "height": "sm",
                    "color": "#905c44",
                    "action": {
                        "type": "uri",
                        "label": "ดูสินค้า",
                        "uri": prod['url']
                    }
                }
            ],
            "flex": 0
        }
    } for prod in products]

    # สร้าง Flex Message content
    contents = {"type": "carousel", "contents": bubbles}

    flex_message = FlexSendMessage(
        alt_text="Best Selling Products",
        contents=contents
    )

    # ตอบกลับด้วย Flex Message
    line_bot_api.reply_message(reply_token, [
        flex_message,
        TextSendMessage(text="นี่คือสินค้าที่แนะนำ", quick_reply=main_quick_reply())
    ])

# Function to send Flex Message with product details
# Function to send Flex Message with product details
def send_flex_message(reply_token, products):
    if not products:
        text_message = TextSendMessage(
            text="ไม่พบสินค้าตามที่ค้นหา.", 
            quick_reply=main_quick_reply()  # เพิ่ม Quick Reply เมื่อไม่พบสินค้า
        )
        line_bot_api.reply_message(reply_token, text_message)
        return

    # Generate Flex Message bubbles with button linking to the product URL
    bubbles = [{
        "type": "bubble",
        "hero": {
            "type": "image",
            "url": prod['image'],  # Use absolute URL for the image
            "size": "full",
            "aspectRatio": "20:13",
            "aspectMode": "cover"
        },
        "body": {
            "type": "box",
            "layout": "vertical",
            "contents": [
                {"type": "text", "text": prod['name'], "weight": "bold", "size": "md", "wrap": True},
                {"type": "text", "text": f"Price: {prod['price']}", "size": "sm", "color": "#999999"}
            ]
        },
        "footer": {
            "type": "box",
            "layout": "vertical",
            "spacing": "sm",
            "contents": [
                {
                    "type": "button",
                    "style": "primary",  # Changed to 'primary' for colored button
                    "height": "sm",
                    "color": "#905c44",  # You can change this color to any color you like
                    "action": {
                        "type": "uri",
                        "label": "ดูสินค้า",
                        "uri": prod['url']  # Link to product page
                    }
                }
            ],
            "flex": 0
        }
    } for prod in products]

    # Create Flex Message content
    contents = {"type": "carousel", "contents": bubbles}

    flex_message = FlexSendMessage(
        alt_text="Product List",
        contents=contents
    )

    # Reply with Flex Message and include additional message about product options
    line_bot_api.reply_message(reply_token, [
        flex_message,
        TextSendMessage(
            text="นี่คือสินค้าที่คุณสนใจ สามารถกดดูสินค้า หรือเลือกดูสินค้าอื่นๆต่อไปได้ครับ",  # ข้อความที่ต้องการแสดงเพิ่ม
            quick_reply=main_quick_reply()  # เพิ่ม Quick Reply ให้ผู้ใช้เลือกต่อ
        )
    ])

# Function to save chat history to Neo4j
def save_chat_history(user_id, user_message, bot_reply):
    with driver.session() as session:
        # Create or update the user with their message and bot's reply in Neo4j
        query = (
            "MERGE (u:User {user_id: $user_id}) "  # Create user node if it doesn't exist
            "ON CREATE SET u.created_at = timestamp() "  # Set creation time when user is first created
            "SET u.last_message = $message, u.last_reply = $reply, u.updated_at = timestamp() "  # Update message and timestamp
            "CREATE (c:Chat {message: $message, reply: $reply, timestamp: timestamp()}) "  # Create a new chat node for this interaction
            "MERGE (u)-[:HAS_CHAT]->(c) "  # Link the user to their chat history
        )
        session.run(query, user_id=user_id, message=user_message, reply=bot_reply)


# ฟังก์ชันแปลงข้อความภาษาไทยเป็นหมวดหมู่สินค้า
def translate_to_english(thai_text):
    translation_map = {
        "เสื้อ": "Shirts",
        "กระเป๋า": "Tote Bag",
        "ถุงเท้า": "Socks",
        "กางเกง": "Pants",
        "หมวก": "Hats",
        "รองเท้า": "Shoe"
    }
    return translation_map.get(thai_text, None)

# ปรับปรุง linebot() เพื่อให้บอทตอบสนองต่อชื่อภาษาไทย
@app.route("/", methods=['POST'])
def linebot():
    body = request.get_data(as_text=True)
    try:
        json_data = json.loads(body)
        reply_token = json_data['events'][0]['replyToken']
        user_id = json_data['events'][0]['source']['userId']
        msg = json_data['events'][0]['message']['text']

        # ตรวจสอบว่าผู้ใช้พิมพ์ข้อความเฉพาะเพื่อเรียก main_quick_reply
        if msg.lower() in ["menu", "เมนู", "main", "quick reply"]:  # เพิ่มคำที่ต้องการให้ตอบกลับด้วย quick reply
            bot_reply = "เลือกหมวดหมู่ที่คุณสนใจ"
            line_bot_api.reply_message(reply_token, [
                TextSendMessage(text=bot_reply, quick_reply=main_quick_reply())
            ])
            save_chat_history(user_id, msg, bot_reply)
            return 'OK'

        # ตรวจสอบว่าข้อความเป็นคำภาษาไทยและแปลงเป็นหมวดหมู่ภาษาอังกฤษ
        translated_category = translate_to_english(msg)
        if translated_category:
            # สินค้าที่แปลงจากภาษาไทยเป็นภาษาอังกฤษได้ (Shirts, Tote Bag, etc.)
            products = scrape_converse(translated_category)
            send_flex_message(reply_token, products)
            bot_reply = f"แสดงสินค้าสำหรับ {translated_category}"
            save_chat_history(user_id, msg, bot_reply)
            return 'OK'

        # แยกเคสของ "Sale!!" ออกจากการตรวจสอบ greeting
        elif msg == "Sale!!":
            bot_reply = "ยังไม่มีรุ่นไหนลดราคา"
            line_bot_api.reply_message(reply_token, TextSendMessage(text=bot_reply, quick_reply=main_quick_reply()))
            save_chat_history(user_id, msg, bot_reply)
            return 'OK'

        # ถามคำแนะนำหรือ Best Selling
        elif msg.lower() in ["ขอคำแนะนำ", "แนะนำ", "recommend", "best selling"]:
            send_best_selling_flex_message(reply_token)
            return 'OK'

        # ตรวจสอบข้อความ greeting ปกติ
        matched_greeting = compute_similar(msg)
        if matched_greeting:
            # Fetch reply for the matched greeting from Neo4j
            bot_reply = neo4j_search(matched_greeting)
            line_bot_api.reply_message(reply_token, [
                TextSendMessage(text=bot_reply),
                TextSendMessage(text="เลือกหมวดหมู่ หรือ รุ่นที่สนใจ หรือ พิมพ์ แนะนำ เพื่อดูสินค้าขายดีได้ครับ", quick_reply=main_quick_reply())
            ])
            save_chat_history(user_id, msg, bot_reply)
            return 'OK'

        # Handle each main category selection (Shoe, Collection, Product)
        if msg == "Shoe":
            quick_reply = QuickReply(items=[
                QuickReplyButton(action=MessageAction(label="RISE COFFEE", text="RISE COFFEE")),
                QuickReplyButton(action=MessageAction(label="MAISON KEEPS", text="MAISON KEEPS")),
                QuickReplyButton(action=MessageAction(label="GAT", text="GAT")),
                QuickReplyButton(action=MessageAction(label="ASTRO", text="ASTRO")),
                QuickReplyButton(action=MessageAction(label="ALEXIS", text="ALEXIS")),
                QuickReplyButton(action=MessageAction(label="BUMPER", text="BUMPER")),
                QuickReplyButton(action=MessageAction(label="COOPER", text="COOPER")),
                QuickReplyButton(action=MessageAction(label="SLIP ON", text="SLIP ON")),
                QuickReplyButton(action=MessageAction(label="MACC", text="MACC")),
                QuickReplyButton(action=MessageAction(label="HI TOP", text="HI TOP"))
            ])
            bot_reply = "สนใจรองเท้ารุ่นไหน? หรือพิมพ์ เมนู เพิ่อกลับไปเลือกหมวดหมู่ได้ครับ"
            line_bot_api.reply_message(reply_token, TextSendMessage(text=bot_reply, quick_reply=quick_reply))
            save_chat_history(user_id, msg, bot_reply)
            return 'OK'

        elif msg == "Collection":
            quick_reply = QuickReply(items=[
                QuickReplyButton(action=MessageAction(label="MAVERICKS", text="MAVERICKS")),
                QuickReplyButton(action=MessageAction(label="ODYSSEE", text="ODYSSEE")),
                QuickReplyButton(action=MessageAction(label="MIDNIGHT SUN", text="MIDNIGHT SUN")),
                QuickReplyButton(action=MessageAction(label="MACC", text="MACC"))
            ])
            bot_reply = "สนใจ Collection ไหนครับ? หรือพิมพ์ เมนู เพิ่อกลับไปเลือกหมวดหมู่ได้ครับ"
            line_bot_api.reply_message(reply_token, TextSendMessage(text=bot_reply, quick_reply=quick_reply))
            save_chat_history(user_id, msg, bot_reply)
            return 'OK'

        elif msg == "Product":
            quick_reply = QuickReply(items=[
                QuickReplyButton(action=MessageAction(label="Shirts", text="Shirts")),
                QuickReplyButton(action=MessageAction(label="Tote Bag", text="Tote Bag")),
                QuickReplyButton(action=MessageAction(label="Socks", text="Socks")),
                QuickReplyButton(action=MessageAction(label="Pants", text="Pants")),
                QuickReplyButton(action=MessageAction(label="Hats", text="Hats"))
            ])
            bot_reply = "สนใจอะไรครับ? หรือพิมพ์ เมนู เพิ่อกลับไปเลือกหมวดหมู่ได้ครับ"
            line_bot_api.reply_message(reply_token, TextSendMessage(text=bot_reply, quick_reply=quick_reply))
            save_chat_history(user_id, msg, bot_reply)
            return 'OK'

        elif msg == "General":
            bot_reply = 'คำถามทั่วไป\n1.ลองสินค้าจริงได้ที่ไหนบ้าง\n2.Mustard Sneakers เป็นแบรนด์ของที่ไหน\n3.รองเท้าทำมาจากวัสดุอะไร\n4.ทำความสะอาดรองเท้าอย่างไร'
            line_bot_api.reply_message(reply_token, [TextSendMessage(text=bot_reply, quick_reply=general_quick_reply())])
            save_chat_history(user_id, msg, bot_reply)
            return 'OK'

        # Handle each numbered FAQ selection
        elif msg in ["1", "2", "3", "4"]:
            faqs = scrape_general_faq()  # Scrape the FAQ answers
            if msg in faqs:
                bot_reply = faqs[msg]  # Get the corresponding answer
                line_bot_api.reply_message(reply_token, [
                    TextSendMessage(text=bot_reply),  # First message: Display the answer
                    TextSendMessage(
                        text='คำถามทั่วไป\n1.ลองสินค้าจริงได้ที่ไหนบ้าง\n2.Mustard Sneakers เป็นแบรนด์ของที่ไหน\n3.รองเท้าทำมาจากวัสดุอะไร\n4.ทำความสะอาดรองเท้าอย่างไร', 
                        quick_reply=general_quick_reply()  # Second message: Show the "General" menu again
                    )
                ])
            else:
                bot_reply = "ขออภัย ไม่พบคำตอบสำหรับคำถามนี้"
                line_bot_api.reply_message(reply_token, [
                    TextSendMessage(text=bot_reply),  # Error message
                    TextSendMessage(
                        text='คำถามทั่วไป\n1.ลองสินค้าจริงได้ที่ไหนบ้าง\n2.Mustard Sneakers เป็นแบรนด์ของที่ไหน\n3.รองเท้าทำมาจากวัสดุอะไร\n4.ทำความสะอาดรองเท้าอย่างไร', 
                        quick_reply=general_quick_reply()  # Second message: Show the "General" menu again
                    )
                ])
            save_chat_history(user_id, msg, bot_reply)
            return 'OK'

        else:
            # Handle product selections and scrape data
            products = scrape_converse(msg)
            send_flex_message(reply_token, products)
            bot_reply = f"แสดงสินค้าสำหรับ {msg}"

    except Exception as e:
        print(f"Error processing the LINE event: {e}")

    return 'OK'

if __name__ == '__main__':
    app.run(port=5000, debug=True)
