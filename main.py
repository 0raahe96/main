


# في بداية البرنامج


import os
import time
import psutil
import pandas as pd
import numpy as np
import joblib
import aiohttp
import shutil
import glob
import asyncio
import ccxt
import matplotlib.pyplot as plt
import mplfinance as mpf
import warnings
import onnxruntime as ort
import logging
import urllib3
from datetime import datetime
import requests
import re
import feedparser
import signal
import sys
from binance.client import Client as BinanceClient
from binance.enums import *
from functools import lru_cache
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from onnxmltools.convert import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import traceback
import gc
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import tensorflow as tf
from functools import lru_cache
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.compat.v1 import ConfigProto
import pandas_ta as ta
from datetime import timedelta, datetime
import shutil
import asyncio
import subprocess
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    Application
)
import requests
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
# في بداية الكود، تأكد من إعداد التسجيل المناسب
import logging
import sys
from flask import Flask
import threading
import os

app = Flask(__name__)

@app.route("/")
def home():
    return "🤖 البوت يعمل على Render!"

def run_flask():
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)

# تشغيل السيرفر في Thread منفصل
threading.Thread(target=run_flask, daemon=True).start()



# إعداد نظام التسجيل لدعم Unicode
class UnicodeStreamHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            # encode the message to UTF-8 and decode it to handle special characters
            stream.write(msg.encode('utf-8', 'replace').decode('utf-8') + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)


# إعداد التهيئة الأساسية للتسجيل
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_bot.log", encoding='utf-8'),
        UnicodeStreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)



BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
AUTHORIZED_USER_ID = os.getenv("AUTHORIZED_USER_ID")
CHAT_ID = AUTHORIZED_USER_ID 

# ====== تحميل المتغيرات البيئية ======
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, '.env')
load_dotenv(dotenv_path=env_path)

from ccxt.async_support import binance

exchange = binance({
    "apiKey": BINANCE_API_KEY,
    "secret": BINANCE_API_SECRET,
    "enableRateLimit": True,
    "options": {"defaultType": "future"},
})


binance_client = BinanceClient(api_key=BINANCE_API_KEY, api_secret=BINANCE_API_SECRET)


# طباعة المتغيرات للتأكد من تحميلها
print(f"TELEGRAM_TOKEN: {'*****' if TELEGRAM_TOKEN else 'غير معين'}")
print(f"AUTHORIZED_USER_ID: {AUTHORIZED_USER_ID or 'غير معين'}")
print(f"BINANCE_API_KEY: {'*****' if BINANCE_API_KEY else 'غير معين'}")
print(f"BINANCE_SECRET_KEY: {'*****' if BINANCE_API_SECRET else 'غير معين'}")

import asyncio, sys

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

def analyze_sentiment(title):
    """تحليل مشاعر عنوان الخبر"""
    positive_keywords = ['صعود', 'ارتفاع', 'مكسب', 'قفزة', 'ايجابي', 'bullish', 'rise', 'surge', 'gain', 'positive']
    negative_keywords = ['هبوط', 'انخفاض', 'خسارة', 'انهيار', 'سلبي', 'bearish', 'fall', 'crash', 'drop', 'negative']
    
    title_lower = title.lower()
    
    positive_count = sum(title_lower.count(kw) for kw in positive_keywords)
    negative_count = sum(title_lower.count(kw) for kw in negative_keywords)
    
    if positive_count > negative_count:
        return 'إيجابي 🟢'
    elif negative_count > positive_count:
        return 'سلبي 🔴'
    else:
        return 'محايد ⚪️'

class TTLCache:
    def __init__(self, ttl=3600, maxsize=32):
        self.cache = {}
        self.ttl = ttl
        self.maxsize = maxsize
        
    def get(self, key):
        if key in self.cache:
            value, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return value
            else:
                del self.cache[key]
        return None
        
    def set(self, key, value):
        if len(self.cache) >= self.maxsize:
            # إزالة أقدم عنصر
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        self.cache[key] = (value, time.time())

# إنشاء ذاكرة تخزين مؤقتة للأخبار
news_cache = TTLCache(ttl=3600, maxsize=32)

# ====== دالة جلب الأخبار المعدلة ======
# ... (بقية الكود حتى تعريف TTLCache) ...



import socket

def check_internet_connection(host="8.8.8.8", port=53, timeout=3):
    """التحقق من الاتصال بالإنترنت"""
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except socket.error:
        return False

def safe_request(url, headers=None, params=None, timeout=10, max_retries=3):
    """طلب آمن مع إعادة المحاولة ومعالجة الأخطاء المحسنة"""
    # التحقق من الاتصال أولاً
    if not check_internet_connection():
        print("❌ لا يوجد اتصال بالإنترنت")
        return None
        
    headers = headers or {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, params=params, timeout=timeout, verify=False)
            response.raise_for_status()
            return response
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2
                print(f"⚠️ المحاولة {attempt+1}/{max_retries} لـ {url} فشلت")
                time.sleep(wait_time)
            else:
                print(f"❌ فشل جميع المحاولات لـ {url}")
                return None
            
# ====== دالة جلب الأخبار المعدلة ======
def get_crypto_news(coin_symbol, num_articles=3):
    """جلب أخبار العملات مع نظام تجاوز الأخطاء"""
    # التحقق من وجود نسخة مخبأة
    cache_key = f"{coin_symbol}_{num_articles}"
    cached = news_cache.get(cache_key)
    if cached is not None:
        return cached
        
    # قائمة موسعة من مصادر RSS مع أوزان
    rss_sources = [
        {'url': f'https://cryptonews.com/news/{coin_symbol.lower()}/feed/', 'weight': 1.0},
        {'url': f'https://cointelegraph.com/rss/tag/{coin_symbol.lower()}', 'weight': 1.0},
        {'url': 'https://ambcrypto.com/feed/', 'weight': 0.9},
        {'url': 'https://www.coindesk.com/arc/outboundfeeds/rss/?outputType=xml', 'weight': 0.9},
        {'url': 'https://beincrypto.com/feed/', 'weight': 0.8},
        {'url': 'https://cryptopotato.com/feed/', 'weight': 0.8},
        {'url': 'https://www.newsbtc.com/feed/', 'weight': 0.8},
        {'url': 'https://cryptoslate.com/feed/', 'weight': 0.7},
        {'url': 'https://u.today/rss', 'weight': 0.6}  # وزن أقل للأخير
    ]
    
    articles = []
    seen_titles = set()
    
    # ترتيب المصادر حسب الوزن
    rss_sources.sort(key=lambda x: x['weight'], reverse=True)
    
    for source in rss_sources:
        if len(articles) >= num_articles:
            break
            
        try:
            # استخدام نظام الطلبات الآمن
            response = safe_request(source['url'])
            if not response:
                continue
                
            feed = feedparser.parse(response.content)
            
            for entry in feed.entries:
                if len(articles) >= num_articles:
                    break
                    
                # تخطي العناوين الفارغة
                if not hasattr(entry, 'title') or not entry.title:
                    continue
                    
                # منع التكرار
                title_hash = hash(entry.title)
                if title_hash in seen_titles:
                    continue
                    
                # حساب وقت النشر
                pub_time = datetime.utcnow()
                if hasattr(entry, 'published_parsed'):
                    pub_time = datetime(*entry.published_parsed[:6])
                elif hasattr(entry, 'updated_parsed'):
                    pub_time = datetime(*entry.updated_parsed[:6])
                    
                hours_ago = (datetime.utcnow() - pub_time).total_seconds() // 3600
                
                # تجاهل الأخبار القديمة
                if hours_ago > 48:
                    continue
                
                # تحليل مشاعر العنوان
                sentiment = analyze_sentiment(entry.title)
                
                articles.append({
                    "title": entry.title[:120] + '...' if len(entry.title) > 120 else entry.title,
                    "url": entry.link,
                    "source": feed.feed.get('title', source['url'].split('//')[1].split('/')[0]) if hasattr(feed.feed, 'title') else source['url'].split('//')[1].split('/')[0],
                    "hours_ago": int(hours_ago),
                    "sentiment": sentiment
                })
                seen_titles.add(title_hash)
                
        except Exception as e:
            logging.warning("خطأ في معالجة مصدر %s: %s", source['url'], str(e))
    
    # ترتيب الأخبار حسب الأحدث
    articles.sort(key=lambda x: x['hours_ago'])
    
    # تخزين النتيجة في الذاكرة المؤقتة
    result = articles[:num_articles]
    news_cache.set(cache_key, result)
    
    return result

def get_fear_greed_index():
    """جلب مؤشر الخوف والجشع العام لسوق العملات المشفرة"""
    try:
        url = "https://api.alternative.me/fng/?limit=1"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            
            if 'data' in data and len(data['data']) > 0:
                latest = data['data'][0]
                value = int(latest['value'])
                
                # تحديد التصنيف بناءً على القيمة
                if value <= 25:
                    classification = "خوف شديد"
                elif value <= 45:
                    classification = "خوف"
                elif value <= 55:
                    classification = "حيادي"
                elif value <= 75:
                    classification = "جشع"
                else:
                    classification = "جشع شديد"
                
                # تحويل الطابع الزمني
                timestamp = datetime.fromtimestamp(int(latest['timestamp']))
                
                return {
                    'value': value,
                    'classification': classification,
                    'timestamp': timestamp
                }
    except Exception as e:
        logging.error(f"فشل جلب مؤشر الخوف والجشع العام: {str(e)}")
    
    return None

# ====== تعديل دالة مؤشر الخوف والجشع ======
def get_coin_sentiment(coin_symbol, max_retries=3):
    """جلب مؤشر الخوف والجشع الخاص بعملة محددة مع إعادة المحاولة"""
    LUNARCRUSH_API_KEY = "mhbd5eemkpkj2tu0t63tbi7xmed4qos56vvlok"
    
    url = f"https://api.lunarcrush.com/v2?data=assets&key={LUNARCRUSH_API_KEY}&symbol={coin_symbol}&data_points=1"
    
    for attempt in range(max_retries):
        try:
            response = safe_request(url)
            if not response:
                continue
                
            data = response.json()
            
            if 'data' in data and len(data['data']) > 0:
                asset_data = data['data'][0]
                sentiment_score = asset_data.get('s', 50)
                galaxy_score = asset_data.get('g', 50)
                composite_score = (sentiment_score + galaxy_score) / 2
                
                if composite_score <= 25:
                    classification = "خوف شديد"
                elif composite_score <= 45:
                    classification = "خوف"
                elif composite_score <= 55:
                    classification = "حيادي"
                elif composite_score <= 75:
                    classification = "جشع"
                else:
                    classification = "جشع شديد"
                
                return {
                    'value': round(composite_score, 1),
                    'classification': classification,
                    'sentiment_score': sentiment_score,
                    'galaxy_score': galaxy_score,
                    'timestamp': datetime.utcnow(),
                    'source': 'LunarCrush'
                }
                
        except Exception as e:
            logging.warning(f"المحاولة {attempt+1}: فشل جلب مؤشر المشاعر للعملة {coin_symbol} - {str(e)}")
    
    # نظام بديل إذا فشلت جميع المحاولات
    return get_alternative_sentiment(coin_symbol)



def get_alternative_sentiment(coin_symbol):
    """نظام بديل لجلب مشاعر العملة عند فشل المصدر الرئيسي"""
    try:
        # 1. استخدام بيانات من Binance
        exchange = ccxt.binance({
            "apiKey": BINANCE_API_KEY,
            "secret": BINANCE_API_SECRET,
            "enableRateLimit": True
        })
        
        ticker = exchange.fetch_ticker(f"{coin_symbol}/USDT")
        price_change = ticker['percentage']
        
        # تحويل التغير السعري إلى مؤشر مشاعر
        if price_change < -5:
            sentiment = "خوف شديد"
            score = 20
        elif price_change < 0:
            sentiment = "خوف"
            score = 40
        elif price_change < 5:
            sentiment = "حيادي"
            score = 50
        elif price_change < 10:
            sentiment = "جشع"
            score = 70
        else:
            sentiment = "جشع شديد"
            score = 90
            
        return {
            'value': score,
            'classification': sentiment,
            'sentiment_score': score,
            'galaxy_score': 50,
            'timestamp': datetime.utcnow(),
            'source': 'Binance Price Change'
        }
        
    except Exception as e:
        logging.warning(f"فشل النظام البديل: {str(e)}")
        return {
            'value': 50,
            'classification': "حيادي",
            'sentiment_score': 50,
            'galaxy_score': 50,
            'timestamp': datetime.utcnow(),
            'source': 'Default'
        }
# ====== وظائف مساعدة للذاكرة المؤقتة ======
def model_cache(func):
    """مزود للذاكرة المؤقتة للنماذج"""
    cache = {}
    
    @wraps(func)
    def wrapper(model_path, *args, **kwargs):
        if model_path in cache:
            return cache[model_path]
        
        result = func(model_path, *args, **kwargs)
        cache[model_path] = result
        return result
    
    return wrapper

def clean_old_cache(model_dir, max_age_days=3):
    """حذف الملفات المؤقتة القديمة"""
    now = time.time()
    cache_dir = os.path.join(model_dir, "cache")
    if not os.path.exists(cache_dir):
        return
        
    for filename in os.listdir(cache_dir):
        filepath = os.path.join(cache_dir, filename)
        if os.path.isfile(filepath):
            file_age = now - os.path.getmtime(filepath)
            if file_age > max_age_days * 86400:
                try:
                    os.remove(filepath)
                except:
                    pass

def enhance_trading_signals(symbol, exchange, predictions, df, current_accuracy, time_frame, base_position_size=0.1):
    """
    دالة لتحسين دقة إشارات التداول من خلال تطبيق مرشحات متعددة
    وتحليل السياق السوقي وتعديل أحجام الصفقات dynamically
    
    المدخلات:
    - symbol: زوج التداول (مثال: 'BTC/USDT')
    - exchange: كائن الاتصال بالبورصة (مثال: ccxt.binance())
    - predictions: قائمة التنبؤات من جميع النماذج
    - df: بيانات الإطار الزمني الحالي
    - current_accuracy: دقة النموذج الحالية
    - time_frame: الإطار الزمني الحالي
    - base_position_size: حجم الصفقة الأساسي (افتراضي: 0.1)
    
    المخرجات:
    - قاموس يحتوي على الإشارة المحسنة ومعلومات إضافية
    """
    
    # 1. مرشح قوة الاتجاه (محسّن)
    def is_strong_trend(df, min_strength=0.01):  # تغيير من 0.02 إلى 0.01
        """التحقق من وجود تحرك قوي في السعر"""
        if len(df) < 5:
            return False
        recent_gain = (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5]
        return abs(recent_gain) >= min_strength
    
    # 2. تحليل السياق السوقي
    def get_market_context(symbol, exchange):
        """تحليل الاتجاه العام وقوة السوق"""
        try:
            # جلب بيانات الإطار اليومي للاتجاه العام
            daily_candles = exchange.fetch_ohlcv(symbol, timeframe='1d', limit=50)
            daily_df = pd.DataFrame(daily_candles, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
            
            # حساب المتوسطات المتحركة
            daily_df['sma_20'] = daily_df['close'].rolling(20).mean()
            daily_df['sma_50'] = daily_df['close'].rolling(50).mean()
            
            # تحديد الاتجاه العام
            current_close = daily_df['close'].iloc[-1]
            sma_20 = daily_df['sma_20'].iloc[-1]
            sma_50 = daily_df['sma_50'].iloc[-1]
            
            # تحديد الاتجاه بناء على المتوسطات
            if current_close > sma_20 and current_close > sma_50:
                trend_direction = "صعود"
            elif current_close < sma_20 and current_close < sma_50:
                trend_direction = "هبوط"
            else:
                trend_direction = "محايد"
            
            # حساب قوة الاتجاه (نسبة الانحراف عن المتوسط)
            deviation_20 = abs(current_close - sma_20) / sma_20 if sma_20 != 0 else 0
            deviation_50 = abs(current_close - sma_50) / sma_50 if sma_50 != 0 else 0
            context_strength = (deviation_20 + deviation_50) / 2
            
            return context_strength, trend_direction
            
        except Exception as e:
            print(f"Error in market context analysis: {e}")
            return 0.5, "محايد"  # القيم الافتراضية في حالة الخطأ
    
    # 3. تأكيد متعدد الأطر الزمنية (محسّن)
    def requires_multi_timeframe_confirmation(predictions, min_confirmations=2):  # تغيير من 3 إلى 2
        """التأكد من وجود تأكيد من عدة أطر زمنية"""
        if not predictions or len(predictions) < 2:
            return False
            
        # حساب عدد التوقعات المتوافقة مع الإتجاه الأول
        first_direction = predictions[0]['final_direction']
        confirmations = sum(1 for p in predictions if p['final_direction'] == first_direction)
        
        return confirmations >= min_confirmations
    
    # 4. إدارة المخاطر الديناميكية
    def dynamic_position_size(confidence, volatility, base_size=0.1):
        """تعديل حجم الصفقة بناء على قوة الإشارة والتقلبات"""
        if confidence < 0.6:
            return base_size * 0.3  # تقليل كبير للحجم
        elif confidence < 0.7:
            return base_size * 0.5  # تقليل حجم الصفقة
        elif volatility > 0.05:
            return base_size * 0.7  # تقليل الحجم في التقلبات العالية
        elif confidence > 0.8:
            return base_size * 1.2  # زيادة الحجم للإشارات عالية الثقة
        else:
            return base_size
    
    # 5. إعادة التدريب التلقائي
    def auto_retrain_model(symbol, time_frame, performance_threshold=0.55):
        """إعادة تدريب النموذج تلقائياً عند انخفاض الأداء"""
        if current_accuracy < performance_threshold:
            retrain_msg = f"نموذج {symbol} على الإطار {time_frame} يحتاج إعادة تدريب (الدقة: {current_accuracy:.2f})"
            print(retrain_msg)
            
            # إرسال إشعار إلى Telegram
            try:
                send_telegram_message(retrain_msg)
            except:
                pass  # تجاهل الخطأ إذا لم تكن الدالة متاحة
            
            return True
        return False
    
    # تطبيق التحسينات
    enhanced_signal = None
    confidence = predictions[0]['confidence'] if predictions and len(predictions) > 0 else 0.5
    
    # حساب التقلبات الأخيرة
    recent_volatility = df['close'].pct_change().std() if len(df) > 1 else 0.02
    
    # 1. تطبيق مرشح قوة الاتجاه
    strong_trend = is_strong_trend(df)
    
    # 2. تحليل السياق السوقي
    market_context, overall_trend = get_market_context(symbol, exchange)
    
    # 3. التأكد من تعدد التأكيدات
    multi_tf_confirmation = requires_multi_timeframe_confirmation(predictions)
    
    # 4. إعادة التدريب إذا لزم الأمر
    needs_retraining = auto_retrain_model(symbol, time_frame)
    
    # نظام النقاط المرن لتحديد قوة الإشارة
    points = 0
    original_signal = predictions[0]['final_direction'] if predictions and len(predictions) > 0 else "محايد"
    
    # منح النقاط بناء على المعايير
    if original_signal == overall_trend:
        points += 2  # +2 نقطة للتطابق مع اتجاه السوق
    
    if multi_tf_confirmation:
        points += 2  # +2 نقطة للتأكيد المتعدد
    
    if strong_trend:
        points += 1  # +1 نقطة لوجود اتجاه قوي
    
    if confidence > 0.7:
        points += 1  # +1 نقطة لثقة عالية
    
    # تحديد الإشارة النهائية بناء على النقاط
    if points >= 4:  # 4-6 نقاط: إشارة قوية
        enhanced_signal = original_signal
        signal_strength = "قوية"
    elif points >= 2:  # 2-3 نقاط: إشارة متوسطة
        enhanced_signal = original_signal
        signal_strength = "متوسطة"
        # تقليل حجم الصفقة للإشارات متوسطة القوة
        base_position_size = base_position_size * 0.7
    else:  # 0-1 نقطة: إشارة ضعيفة
        enhanced_signal = "محايد"
        signal_strength = "ضعيفة"
    
    # حساب حجم الصفقة المناسب
    position_size = dynamic_position_size(confidence, recent_volatility, base_position_size)
    
    # التشخيص والطباعة
    print(f"نقاط الإشارة: {points}/6")
    print(f"قوة الإشارة: {signal_strength}")
    print(f"الإشارة النهائية: {enhanced_signal}")
    
    # إنشاء نتيجة التحسين
    result = {
        'enhanced_signal': enhanced_signal,
        'position_size': position_size,
        'market_trend': overall_trend,
        'market_strength': market_context,
        'needs_retraining': needs_retraining,
        'strong_trend': strong_trend,
        'multi_tf_confirmed': multi_tf_confirmation,
        'original_signal': original_signal,
        'confidence': confidence,
        'volatility': recent_volatility,
        'signal_points': points,
        'signal_strength': signal_strength
    }
    
    # إرسال نتيجة التحسين إلى Telegram
    try:
        enhanced_message = f"""
🎯 **الإشارة المحسنة بعد التصفية - {symbol}**
📊 الإتجاه الأصلي: {result['original_signal']}
📈 الإتجاه المحسن: {result['enhanced_signal']} ({result['signal_strength']})
💰 حجم الصفقة المقترح: {result['position_size']}
🔰 اتجاه السوق العام: {result['market_trend']}
💪 قوة السوق: {result['market_strength']:.2f}
✅ تأكيد متعدد الأطر: {result['multi_tf_confirmed']}
📶 تحرك قوي: {result['strong_trend']}
🎚️ الثقة: {result['confidence']:.2f}
📊 التقلبات: {result['volatility']:.4f}
🏆 نقاط الإشارة: {result['signal_points']}/6
        """
        send_telegram_message(enhanced_message)
    except Exception as e:
        print(f"Failed to send Telegram message: {e}")
    
    return result


def generate_more_natural_future_candles(last_close, direction="up", num=150, base_volatility=0.01, trend_strength=1.0, recent_data=None, targets=None):
    """
    توليد شموع مستقبلية واقعية مع تحسين التقلبات والمسارات
    """
    if recent_data is not None and len(recent_data) > 100:
        return recent_data.sample(min(num, len(recent_data)), replace=True).reset_index(drop=True)
    
    # تحسين معاملات التقلب والاتجاه
    volatility = base_volatility * (1 + np.random.uniform(0.1, 0.5))
    
    if direction == "up":
        trend_factor = 0.3 + (trend_strength * 0.7)
    elif direction == "down":
        trend_factor = - (0.3 + (trend_strength * 0.7))
    else:
        trend_factor = np.random.uniform(-0.2, 0.2)

    # تحسين تأثير الأهداف
    target_influence = 0
    if targets and len(targets) >= 3:
        avg_target = np.mean(targets)
        target_distance = abs(avg_target - last_close) / last_close
        target_influence = min(0.3, max(0.05, target_distance * 0.5))

    prices = [last_close]
    
    # إضافة نمط أكثر واقعية مع دورات متعددة
    cycle_lengths = [np.random.randint(20, 40), np.random.randint(50, 80)]
    cycle_weights = [0.6, 0.4]
    
    for i in range(num):
        # تأثير دورات متعددة
        cycle_effect = sum(
            w * np.sin(2 * np.pi * i / L) * volatility * 0.4
            for w, L in zip(cycle_weights, cycle_lengths)
        )
        
        # تحسين التقلبات العشوائية
        random_shock = np.random.normal(0, volatility) * (1 + 0.5 * np.random.rand())
        
        # تأثير الاتجاه
        trend_effect = trend_factor * volatility * (1 - i/num)  # يقل مع الوقت
        
        # تأثير الأهداف (يزداد مع الاقتراب من نهاية الفترة)
        if targets and i > num * 0.6:
            current_target_influence = target_influence * (i / num) ** 2
            target_direction = 1 if direction == "up" else -1
            target_effect = target_direction * current_target_influence * volatility
        else:
            target_effect = 0
            
        # تأثير التصحيح
        if i > 10:
            recent_change = (prices[-1] - prices[-5]) / prices[-5]
            if abs(recent_change) > volatility * 2:
                correction_effect = -np.sign(recent_change) * volatility * 0.7
            else:
                correction_effect = 0
        else:
            correction_effect = 0
            
        # حساب السعر الجديد
        price_change = random_shock + trend_effect + target_effect + cycle_effect + correction_effect
        new_price = prices[-1] * (1 + price_change)
        
        # منع الأسعار السالبة
        if new_price <= 0:
            new_price = prices[-1] * 0.99  # انخفاض طفيف بدلاً من السالب
            
        prices.append(new_price)
    
    # بناء DataFrame للشموع
    df = pd.DataFrame({
        "open": prices[:-1],
        "close": prices[1:]
    })
    
    # تحسين توليد High وLow بشكل أكثر واقعية
    for i in range(len(df)):
        open_price = df.iloc[i]["open"]
        close_price = df.iloc[i]["close"]
        body_range = abs(open_price - close_price)
        
        # تحديد مدى الظلال بناء على التقلب
        shadow_ratio = np.random.uniform(0.1, 0.8)
        upper_shadow = body_range * shadow_ratio * np.random.uniform(0.5, 2)
        lower_shadow = body_range * shadow_ratio * np.random.uniform(0.5, 2)
        
        # تطبيق تأثير الاتجاه على الظلال
        if direction == "up":
            upper_shadow *= 1.2
            lower_shadow *= 0.8
        elif direction == "down":
            upper_shadow *= 0.8
            lower_shadow *= 1.2
            
        # إضافة تأثير العشوائية
        noise_factor = np.random.uniform(0.8, 1.2)
        upper_shadow *= noise_factor
        lower_shadow *= noise_factor
        
        # تعيين High وLow
        df.loc[i, "high"] = max(open_price, close_price) + upper_shadow
        df.loc[i, "low"] = min(open_price, close_price) - lower_shadow
        
        # التأكد من أن High >= Max(Open, Close) و Low <= Min(Open, Close)
        df.loc[i, "high"] = max(df.loc[i, "high"], max(open_price, close_price))
        df.loc[i, "low"] = min(df.loc[i, "low"], min(open_price, close_price))
    
    # تحسين توليد الحجم
    base_volume = np.random.normal(100, 20, size=num)
    
    # ربط الحجم بحركة السعر
    price_changes = df["close"].pct_change().abs()
    volume_multiplier = 1 + price_changes * 10
    volume_multiplier.iloc[0] = 1
    
    # إضافة نمط حجم دوري
    volume_pattern = 0.7 + 0.3 * np.sin(2 * np.pi * np.arange(num) / 30)
    
    df["volume"] = base_volume * volume_multiplier * volume_pattern
    
    # تطبيع الحجم لمنع القيم المتطرفة
    df["volume"] = df["volume"].clip(lower=50, upper=200)
    
    return df



# ====== الدوال الرئيسية ======
def get_signal(symbol, model_dir="models"):
    os.makedirs(model_dir, exist_ok=True)
    signal_data = None
    clean_old_cache(model_dir)  # تنظيف الذاكرة المؤقتة القديمة

# إضافة هذه المتغيرات الافتراضية في بداية الدالة
    overall_direction = "غير محدد"
    tp1_global = tp2_global = tp3_global = 0
    global_sl = 0
    avg_confidence = 0
    
    # ====== إعداد نظام التسجيل (Logging) ======
    log_file = os.path.join(model_dir, "trading_bot.log")
    logging.basicConfig(
        filename=log_file,
        level=logging.ERROR,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    warnings.filterwarnings("ignore")
   # ====== دالة مساعدة لتنظيف الذاكرة ======
    def clear_keras_memory():
        tf.keras.backend.clear_session()
        gc.collect()
   


    # تفعيل نمو الذاكرة التلقائي للـ GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"Error setting memory growth: {e}")
        
        # تحديد نسبة الذاكرة المستخدمة (اختياري)
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=480)]  # 6GB
            )
        except RuntimeError as e:
            print(f"Error setting memory limit: {e}")


 

    # ====== إعدادات التداول ======
    BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
    BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY")
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
    AUTHORIZED_USER_ID = os.getenv("AUTHORIZED_USER_ID")
    
    # إرسال إشعار بدء التحليل
    send_telegram_message(f"🔍 بدء تحليل {symbol}...")
    
    exchange = ccxt.binance({
        "apiKey": BINANCE_API_KEY,
        "secret": BINANCE_SECRET_KEY,
        "enableRateLimit": True,
        "options": {"defaultType": "future"},
    })
    
    # ====== تعريف الدوال المساعدة ======
    def retry_api_call(func, max_retries=5, delay=2, backoff=2):
        """إعادة المحاولة التلقائية لاستدعاءات API"""
        retries = 0
        while retries < max_retries:
            try:
                return func()
            except Exception as e:
                retries += 1
                sleep_time = delay * (backoff ** (retries - 1))
                logger.error(f"API call failed (attempt {retries}/{max_retries}): {str(e)}")
                time.sleep(sleep_time)
        logger.error(f"API call failed after {max_retries} attempts")
        return None

    @lru_cache(maxsize=4)
    def get_cached_data(symbol, time_frame, since):
        """التخزين المؤقت للبيانات لتجنب إعادة الجلب"""
        cache_key = f"{symbol}_{time_frame}_{since}"
        cache_dir = os.path.join(model_dir, "cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"cache_{cache_key}.pkl")
        
        # محاولة تحميل البيانات من الذاكرة المؤقتة
        if os.path.exists(cache_file):
            try:
                data = pd.read_pickle(cache_file)
                if not data.empty:
                    return data
            except:
                pass
        
        # جلب البيانات من API إذا لم توجد في الذاكرة المؤقتة
        def fetch_data():
            return exchange.fetch_ohlcv(symbol, timeframe=time_frame, since=since, limit=2000)
        
        candles = retry_api_call(lambda: fetch_data())
        if not candles:
            return pd.DataFrame()
        
        # تحويل البيانات وتخزينها
        df = pd.DataFrame(candles, columns=["time", "open", "high", "low", "close", "volume"])
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        df.to_pickle(cache_file)
        return df

    def get_real_time_price(symbol):
        """استرجاع السعر الحالي الفعلي للعملة من مصدر موثوق"""
        try:
            ticker = retry_api_call(lambda: exchange.fetch_ticker(symbol))
            return ticker['last'] if ticker else None
        except Exception as e:
            logger.error(f"Error getting real-time price: {traceback.format_exc()}")
            return None

    def get_base_tp_ratios(time_frame):
        mapping = {
            "1m": [0.003, 0.006, 0.009],
            "3m": [0.006, 0.012, 0.018],
            "5m": [0.009, 0.018, 0.027],
            "15m": [0.015, 0.03, 0.045],
            "30m": [0.025, 0.05, 0.075],
        }
        return mapping.get(time_frame, [0.005, 0.01, 0.015])

    def smart_targets_sl(price, direction, confidence, volatility=0.01, base_tp_ratios=None, key_levels=None):
        if base_tp_ratios is None:
            base_tp_ratios = [0.005, 0.01, 0.015]

        # حساب نسب جني الأرباح الديناميكية
        adjusted_ratios = [
            ratio * (0.8 + confidence * 0.4) * (1 + min(volatility, 0.05) * 5)
            for ratio in base_tp_ratios
        ]

        # حساب وقف الخسارة باستخدام المستويات إذا كانت متاحة
        if key_levels:
            stop_loss = calculate_smart_stop_loss(
                "up" if direction == "up" else "down",
                price,
                key_levels,
                volatility
            )
        else:
            # استخدام الطريقة القديمة كاحتياطي
            sl_ratio = 0.007 * (1.5 - confidence * 0.5) * (1 + min(volatility, 0.05) * 3)
            if direction == "up":
                stop_loss = round(price * (1 - sl_ratio), 6)
            else:
                stop_loss = round(price * (1 + sl_ratio), 6)

        tp_targets = []
        for ratio in adjusted_ratios:
            if direction == "up":
                tp_targets.append(round(price * (1 + ratio), 6))
            else:
                tp_targets.append(round(price * (1 - ratio), 6))

        return tp_targets, stop_loss


    def avg_confidence(predicts, keys):
        filtered = [p for p in predicts if p.get("time_frame") in keys]
        if not filtered:
            return 0
        return sum(p.get("confidence", 0) for p in filtered) / len(filtered)

    
    # ====== تحسين بنية نموذج LSTM ======
    def build_optimized_lstm_model(input_shape, num_classes=2):
        """نموذج LSTM مُحسّن للأداء"""
        model = Sequential()
        model.add(LSTM(128, input_shape=input_shape, return_sequences=True))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        
        model.add(LSTM(64, return_sequences=True))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        
        model.add(LSTM(32))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        
        model.add(Dense(16, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
        
        optimizer = Adam(learning_rate=0.001, clipvalue=0.5)
        model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    # ====== تحسين معالجة البيانات ======
    def create_lstm_dataset_optimized(X, y, time_steps=30):
        """إنشاء مجموعة بيانات للتدريب بشكل أكثر كفاءة"""
        Xs, ys = [], []
        for i in range(time_steps, len(X)):
            v = X.iloc[i-time_steps:i].values
            Xs.append(v)
            ys.append(y.iloc[i])
        
        return np.array(Xs), np.array(ys)

    # ====== تحسين التنبؤ المتعدد الخطوات ======
    def efficient_future_prediction(model, last_sequence, steps=10, feature_index=0):
        """تنبؤ أكثر كفاءة بالمستقبل"""
        predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(steps):
            pred = model.predict(current_sequence, verbose=0, batch_size=1)
            pred_value = pred[0][0]
            
            predictions.append(pred_value)
            
            new_step = np.zeros((1, 1, current_sequence.shape[2]))
            new_step[0, 0, feature_index] = pred_value
            
            current_sequence = np.concatenate(
                [current_sequence[:, 1:, :], new_step],
                axis=1
            )
        
        return np.array(predictions)

    def get_dynamic_threshold(time_frame):
        thresholds = {
            "1m": 0.005,
            "3m": 0.008,
            "5m": 0.01,
            "15m": 0.015,
            "30m": 0.02,
        }
        return thresholds.get(time_frame, 0.01)
    
    def clean_and_filter_data(df):
        numeric_cols = ["open", "high", "low", "close", "volume"]
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        df.dropna(inplace=True)
        return df
    
    def add_technical_indicators(df):
        df["rsi"] = ta.rsi(df["close"], length=14)
        macd = ta.macd(df["close"])
        df["macd"] = macd["MACD_12_26_9"]
        df["macdsignal"] = macd["MACDs_12_26_9"]
        stoch = ta.stoch(df["high"], df["low"], df["close"])
        df["stoch_k"] = stoch["STOCHk_14_3_3"]
        df["stoch_d"] = stoch["STOCHd_14_3_3"]
        df["ema_50"] = ta.ema(df["close"], length=50)
        df["ema_200"] = ta.ema(df["close"], length=200)
        adx = ta.adx(df["high"], df["low"], df["close"])
        df["adx"] = adx["ADX_14"]
        df["atr"] = ta.atr(df["high"], df["low"], df["close"])
        df["cci"] = ta.cci(df["high"], df["low"], df["close"])
        df["willr"] = ta.willr(df["high"], df["low"], df["close"])
        return df
    
    def add_candlestick_patterns(df):
        def is_hammer(row):
            body = abs(row["close"] - row["open"])
            lower_shadow = row["open"] - row["low"] if row["close"] > row["open"] else row["close"] - row["low"]
            upper_shadow = row["high"] - max(row["close"], row["open"])
            return int(lower_shadow > 2 * body and upper_shadow < body)
        
        def is_shooting_star(row):
            body = abs(row["close"] - row["open"])
            upper_shadow = row["high"] - max(row["close"], row["open"])
            lower_shadow = min(row["close"], row["open"]) - row["low"]
            return int(upper_shadow > 2 * body and lower_shadow < body)
        
        def is_doji(row):
            return int(abs(row["close"] - row["open"]) <= (row["high"] - row["low"]) * 0.1)
        
        def is_bullish_engulfing(curr, prev):
            return int(
                prev["close"] < prev["open"]
                and curr["close"] > curr["open"]
                and curr["close"] > prev["open"]
                and curr["open"] < prev["close"]
            )
        
        def is_bearish_engulfing(curr, prev):
            return int(
                prev["close"] > prev["open"]
                and curr["close"] < curr["open"]
                and curr["open"] > prev["close"]
                and curr["close"] < prev["open"]
            )
        
        patterns = {
            "hammer": [], "shooting_star": [], "doji": [],
            "bullish_engulfing": [], "bearish_engulfing": []
        }
        
        for i in range(len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i - 1] if i > 0 else row
            patterns["hammer"].append(is_hammer(row))
            patterns["shooting_star"].append(is_shooting_star(row))
            patterns["doji"].append(is_doji(row))
            patterns["bullish_engulfing"].append(is_bullish_engulfing(row, prev_row))
            patterns["bearish_engulfing"].append(is_bearish_engulfing(row, prev_row))
        
        for name in patterns:
            df[name] = patterns[name]
        
        return df
    
    def add_advanced_features(df):
        df = df.copy()
        df['return_1'] = df['close'].pct_change(1)
        df['return_3'] = df['close'].pct_change(3)
        df['return_5'] = df['close'].pct_change(5)
        df['momentum_3'] = df['close'] - df['close'].shift(3)
        df['momentum_5'] = df['close'] - df['close'].shift(5)
        df['volatility_5'] = df['high'].rolling(window=5, min_periods=1).max() - df['low'].rolling(window=5, min_periods=1).min()
        df['volatility_10'] = df['high'].rolling(window=10, min_periods=1).max() - df['low'].rolling(window=10, min_periods=1).min()
        df['volume_price_change'] = df['volume'] * df['return_1']
        df['stddev_5'] = df['close'].rolling(window=5, min_periods=1).std()
        df['stddev_10'] = df['close'].rolling(window=10, min_periods=1).std()
        df['ema_10'] = df['close'].ewm(span=10, min_periods=1, adjust=False).mean()
        df['ema_20'] = df['close'].ewm(span=20, min_periods=1, adjust=False).mean()
        df['ema_diff_10_20'] = df['ema_10'] - df['ema_20']
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
        df.dropna(inplace=True)
        return df
    
    def rename_columns(df):
        df_renamed = df.copy()
        df_renamed.columns = [f"f{i}" for i in range(df_renamed.shape[1])]
        return df_renamed

    # ====== بداية التنفيذ الرئيسي ======
    # قائمة الأطر الزمنية مع تغيير الاسم لتجنب التعارض
    time_frames_list = ["1m", "3m", "5m", "15m", "30m"]
    results = []
    predictions = []
    window_size_for_prediction = 30

    # الحصول على السعر الحالي الفعلي للعملة
    current_real_time_price = get_real_time_price(symbol)
    if current_real_time_price is None:
        results.append("⚠️ تعذر الحصول على السعر الحالي الفعلي، سيتم استخدام سعر الإغلاق الأخير")

    # ====== جلب البيانات باستخدام التخزين المؤقت والتزامن ======
    def fetch_timeframe_data(time_frame):
        try:
            csv_path = os.path.join(model_dir, f"data_{symbol.replace('/', '_')}_{time_frame}.csv")
            
            # جلب البيانات من الذاكرة المؤقتة أو API
            if os.path.exists(csv_path):
                old_df = pd.read_csv(csv_path, parse_dates=["time"])
                since_time = old_df["time"].iloc[-1] - pd.Timedelta(days=3)
                since = int(since_time.timestamp() * 1000)
            else:
                old_df = pd.DataFrame()
                since = int((datetime.now() - timedelta(days=90)).timestamp() * 1000)
            
            # استخدام التخزين المؤقت
            new_data = get_cached_data(symbol, time_frame, since)
            
            if new_data.empty:
                return time_frame, pd.DataFrame(), "❌ فشل جلب أي شموع من البورصة"
                
            # معالجة البيانات
            if not old_df.empty:
                combined_df = pd.concat([old_df, new_data], ignore_index=True)
                combined_df.drop_duplicates(subset="time", inplace=True)
            else:
                combined_df = new_data
                
            combined_df.sort_values("time", inplace=True)
            combined_df.to_csv(csv_path, index=False)
            
            return time_frame, combined_df, "✅ تم تحديث البيانات بنجاح"
            
        except Exception as e:
            logger.error(f"Error processing {time_frame}: {traceback.format_exc()}")
            return time_frame, pd.DataFrame(), f"❌ خطأ في معالجة البيانات: {str(e)}"

    # جلب البيانات باستخدام التزامن
    timeframe_data = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_tf = {executor.submit(fetch_timeframe_data, time_frame): time_frame for time_frame in time_frames_list}
        for future in as_completed(future_to_tf):
            time_frame = future_to_tf[future]
            try:
                time_frame, df, message = future.result()
                timeframe_data[time_frame] = df
                results.append(f"{time_frame}: {message}")
            except Exception as e:
                logger.error(f"Error in thread for {time_frame}: {traceback.format_exc()}")
                results.append(f"❌ خطأ جسيم في معالجة {time_frame}")

    # ====== معالجة كل إطار زمني ======
    for time_frame in time_frames_list:
        try:
            if time_frame not in timeframe_data or timeframe_data[time_frame].empty:
                results.append(f"❌ لا توجد بيانات للفريم {time_frame}")
                continue
                
            # استخدام نسخة عميقة لمنع تعديل البيانات الأصلية
            df = timeframe_data[time_frame].copy(deep=True)
            df = clean_and_filter_data(df)
            df = df[["time", "open", "high", "low", "close", "volume"]]

            if df.empty or len(df) < 100:
                results.append(f"❌ لا توجد بيانات كافية للفريم {time_frame} ({len(df)} صفوف).")
                continue

            onnx_model_path = os.path.join(model_dir, f"model_{symbol.replace('/', '_')}_{time_frame}.onnx")
            pkl_model_path = os.path.join(model_dir, f"model_{symbol.replace('/', '_')}_{time_frame}.pkl")
            lstm_model_path = os.path.join(model_dir, f"model_{symbol.replace('/', '_')}_{time_frame}_lstm.h5")

            # إضافة المؤشرات الفنية وأنماط الشموع
            df = add_technical_indicators(df)
            df = add_candlestick_patterns(df)
            df = add_advanced_features(df)
            
            if df.empty:
                results.append(f"❌ لا توجد بيانات كافية بعد إضافة المؤشرات والميزات في الفريم {time_frame}.")
                continue

            feature_cols = [
                "rsi", "macd", "macdsignal", "stoch_k", "stoch_d", "close", "ema_50", "ema_200", 
                "adx", "atr", "cci", "willr", "hammer", "shooting_star", "bullish_engulfing", 
                "bearish_engulfing", "doji", "return_1", "return_3", "return_5", "momentum_3", 
                "momentum_5", "volatility_5", "volatility_10", "volume_price_change"
            ]
            feature_cols = [col for col in feature_cols if col in df.columns]
            
            if not feature_cols:
                results.append(f"❌ لا توجد ميزات صالحة للفريم {time_frame}.")
                continue

            # ====== إنشاء إشارات متعددة الخطوات ======
            threshold = get_dynamic_threshold(time_frame)
            
            # إشارة قصيرة المدى (شمعة واحدة)
            df['price_change_short'] = df['close'].pct_change(periods=30).shift(30)
            df["signal_short"] = np.where(
                df['price_change_short'] > threshold, 1,  # صعود
                np.where(
                    df['price_change_short'] < -threshold, 0,  # هبوط
                    2  # حيادي
                )
            )
            
            # إشارة متوسطة المدى (10 شمعات)
            df['price_change_medium'] = df['close'].pct_change(periods=10).shift(10)
            df["signal_medium"] = np.where(
                df['price_change_medium'] > threshold * 2, 1,  # صعود
                np.where(
                    df['price_change_medium'] < -threshold * 2, 0,  # هبوط
                    2  # حيادي
                )
            )
            
            # إشارة مركبة (توفيق بين المدى القصير والمتوسط)
            conditions = [
                (df['signal_short'] == 1) & (df['signal_medium'] == 1),  # تأكيد صعودي قوي
                (df['signal_short'] == 0) & (df['signal_medium'] == 0),  # تأكيد هبوطي قوي
                (df['signal_medium'] == 1),  # توجه صعودي متوسط المدى
                (df['signal_medium'] == 0),  # توجه هبوطي متوسط المدى
                (df['signal_short'] == 1),  # توجه صعودي قصير المدى
                (df['signal_short'] == 0),  # توجه هبوطي قصير المدى
            ]

            choices = [1, 0, 1, 0, 1, 0]  # 1: صعود, 0: هبوط
            df['signal'] = np.select(conditions, choices, default=2)
            
            # استخدام فقط الصفوف مع إشارات صالحة
            df_signals = df[df["signal"].isin([0, 1])].copy()
            
            if len(df_signals) < 30:
                results.append(f"❌ لا توجد إشارات كافية (شراء أو بيع) في الفريم {time_frame}.")
                continue

            X = df_signals[feature_cols]
            y = df_signals["signal"]

            # تقسيم البيانات
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, stratify=y, test_size=0.2, random_state=42
            )

            X_train_renamed = rename_columns(X_train)
            X_test_renamed = rename_columns(X_test)

            # ====== إعادة تدريب النماذج عند انخفاض الدقة ======
            ACCURACY_THRESHOLD = 0.6  # الحد الأدنى للدقة
            
            # تدريب نموذج XGBoost إذا لم يكن موجوداً أو دقته ضعيفة
            retrain_needed = False
            if os.path.exists(pkl_model_path):
                try:
                    best_model, feature_cols, metadata = joblib.load(pkl_model_path)
                    last_accuracy = metadata.get('accuracy', 0)
                    
                    if last_accuracy < ACCURACY_THRESHOLD:
                        results.append(f"⚠️ دقة النموذج منخفضة ({last_accuracy:.2f}) - إعادة التدريب")
                        retrain_needed = True
                        os.remove(onnx_model_path)
                        os.remove(pkl_model_path)
                except:
                    retrain_needed = True
            
            if retrain_needed or not os.path.exists(onnx_model_path):
                param_grid = {
                    "max_depth": [3, 5, 7],
                    "n_estimators": [100, 200],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "subsample": [0.8, 1],
                }
                xgb = XGBClassifier(eval_metric="logloss", random_state=42)

                cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                grid = GridSearchCV(
                    xgb, param_grid, cv=cv, scoring="f1", n_jobs=-1, verbose=0
                )
                grid.fit(X_train_renamed, y_train)
                best_model = grid.best_estimator_

                y_pred = best_model.predict(X_test_renamed)
                y_proba = best_model.predict_proba(X_test_renamed)[:, 1]

                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                try:
                    roc_auc = roc_auc_score(y_test, y_proba)
                except:
                    roc_auc = 0

                xgb_acc = acc * 100
                xgb_pred_label = "صعود" if round(np.mean(y_pred)) == 1 else "هبوط"

                results.append(f"✅ تم تدريب وتحويل نموذج XGBoost للفريم {time_frame} بنجاح.")
                results.append(f"▶️ تقييم النموذج: دقة={acc:.4f}, F1={f1:.4f}, ROC_AUC={roc_auc:.4f}")

                # حفظ النموذج مع بيانات الأداء
                metadata = {
                    'accuracy': acc,
                    'f1_score': f1,
                    'roc_auc': roc_auc,
                    'last_trained': datetime.now().isoformat()
                }
                joblib.dump((best_model, feature_cols, metadata), pkl_model_path)
                
                # تحويل إلى ONNX
                initial_type = [("float_input", FloatTensorType([None, X_train_renamed.shape[1]]))]
                onnx_model = convert_xgboost(best_model, initial_types=initial_type)
                with open(onnx_model_path, "wb") as f:
                    f.write(onnx_model.SerializeToString())
            else:
                # تحميل النموذج الموجود
                best_model, feature_cols, metadata = joblib.load(pkl_model_path)
                y_pred = best_model.predict(X_test_renamed)
                acc = accuracy_score(y_test, y_pred)
                xgb_acc = acc * 100
                xgb_pred_label = "صعود" if round(np.mean(y_pred)) == 1 else "هبوط"

            # تحميل وتشغيل نموذج ONNX
            session = ort.InferenceSession(onnx_model_path)
            last_rows = df.iloc[-window_size_for_prediction:]
            X_window = last_rows[feature_cols]
            X_window_renamed = rename_columns(X_window)

            input_name = session.get_inputs()[0].name
            input_data = X_window_renamed.values.astype(np.float32)

            preds = session.run(None, {input_name: input_data})

            if len(preds) == 1:
                labels = [int(p) for p in preds[0]]
                probs = [1.0] * len(labels)
            else:
                probs_all = preds[1]
                labels = np.argmax(probs_all, axis=1)
                probs = probs_all[np.arange(len(labels)), labels]

            short_term_avg = round(np.mean(labels))
            avg_conf = np.mean(probs)

            onnx_acc = avg_conf * 100
            onnx_pred_label = "صعود" if short_term_avg == 1 else "هبوط"

            recent_volatility = last_rows["return_1"].std() if "return_1" in last_rows.columns else 0

            # استخدام السعر الحقيقي بدلاً من السعر التاريخي
            if current_real_time_price is not None:
                price = current_real_time_price
            else:
                price = df.iloc[-1]["close"]

            # ====== تدريب وتوقع نموذج LSTM ======
            scaler = MinMaxScaler()

            # تقسيم البيانات أولاً قبل التحجيم
            split_index = int(0.8 * len(df_signals))
            X_train_raw = X.iloc[:split_index]
            X_test_raw = X.iloc[split_index:]
            y_train_lstm = y.iloc[:split_index]
            y_test_lstm = y.iloc[split_index:]

            # تحجيم البيانات
            X_train_scaled = scaler.fit_transform(X_train_raw)
            X_test_scaled = scaler.transform(X_test_raw)

            # تحويل إلى DataFrames للحفاظ على أسماء الأعمدة
            X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train_raw.columns)
            X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test_raw.columns)

            # إنشاء مجموعات بيانات LSTM
            time_steps = 30
            X_train_lstm, y_train_lstm_arr = create_lstm_dataset_optimized(X_train_scaled_df, y_train_lstm, time_steps=time_steps)
            X_test_lstm, y_test_lstm_arr = create_lstm_dataset_optimized(X_test_scaled_df, y_test_lstm, time_steps=time_steps)

            if len(X_train_lstm) == 0:
                results.append(f"⚠️ بيانات غير كافية لتدريب LSTM في الفريم {time_frame}.")
                lstm_pred = 0
                lstm_acc = 0
                medium_trend = "غير محدد"
            else:
                # التحقق من أبعاد البيانات هنا فقط
                if X_train_lstm.ndim != 3:
                    logger.error(f"شكل خاطئ لبيانات LSTM: {X_train_lstm.shape}، المتوقع 3 أبعاد")
                    # إعادة تشكيل البيانات
                    X_train_lstm = X_train_lstm.reshape((X_train_lstm.shape[0], time_steps, -1))
                    X_test_lstm = X_test_lstm.reshape((X_test_lstm.shape[0], time_steps, -1))
                
                history = None
                # تدريب النموذج فقط إذا لم يكن موجوداً أو يحتاج إعادة تدريب
                if not os.path.exists(lstm_model_path) or retrain_needed:
                    input_shape = (X_train_lstm.shape[1], X_train_lstm.shape[2])
                    lstm_model = build_optimized_lstm_model(input_shape)
                    
                    early_stop = EarlyStopping(
                        monitor="val_loss", 
                        patience=5,
                        restore_best_weights=True
                    )
                    
                    checkpoint = ModelCheckpoint(
                        lstm_model_path, 
                        monitor='val_accuracy', 
                        save_best_only=True, 
                        mode='max'
                    )
                    
                    # تحويل التصنيف إلى ترميز one-hot
                    y_train_categorical = to_categorical(y_train_lstm_arr, num_classes=2)
                    y_test_categorical = to_categorical(y_test_lstm_arr, num_classes=2)
                    
                    # استخدام حجم دفعة ديناميكي
                    batch_size = min(32, max(16, len(X_train_lstm) // 10))
                    
                    history = lstm_model.fit(
                        X_train_lstm,
                        y_train_categorical,
                        validation_data=(X_test_lstm, y_test_categorical),
                        epochs=50,
                        batch_size=batch_size,
                        callbacks=[early_stop, checkpoint],
                        verbose=0,
                    )
                else:
                    lstm_model = load_model(lstm_model_path)
                    # إعادة تجميع النموذج لتجنب التحذيرات
                    lstm_model.compile(
                        optimizer=Adam(learning_rate=0.001),
                        loss='categorical_crossentropy',
                        metrics=['accuracy']
                    )
                
                loss = None  # تعريف مسبق
                accuracy = None 
                # تقييم النموذج
                if len(X_test_lstm) > 0:
                    loss, accuracy = lstm_model.evaluate(
                        X_test_lstm,
                        to_categorical(y_test_lstm_arr, num_classes=2),
                        verbose=0
                    )
                    lstm_acc = accuracy * 100
                else:
                    lstm_acc = 0

                # إعداد البيانات للتنبؤ
                last_data = X.iloc[-time_steps:].copy()
                last_data_scaled = scaler.transform(last_data)
                last_sequence = last_data_scaled.reshape(1, time_steps, last_data_scaled.shape[1])
                
                # التنبؤ بالخطوة القصيرة
                lstm_pred_prob = lstm_model.predict(last_sequence, verbose=0)[0]
                lstm_pred = np.argmax(lstm_pred_prob)
                
                # التنبؤ المتوسط (10 شمعات)
                try:
                    # استخدام أول ميزة (العمود 0) كهدف للتنبؤ
                    medium_predictions = efficient_future_prediction(
                        lstm_model, last_sequence, steps=10, feature_index=3
                    )
                    # تحديد الاتجاه بناءً على متوسط التنبؤات
                    medium_term_avg = np.mean(medium_predictions)
                    medium_trend = "صعود" if medium_term_avg > 0.5 else "هبوط"
                except Exception as e:
                    logger.error(f"Error in medium prediction: {str(e)}")
                    medium_trend = "غير محدد"
                
                results.append(
                    f"🧠 تقييم LSTM للفريم {time_frame}: دقة={lstm_acc:.2f}%"
                )

                if history is not None:
                   train_acc = history.history['accuracy'][-1] * 100
                   val_acc = history.history['val_accuracy'][-1] * 100
                   results.append(f"   → دقة التدريب: {train_acc:.2f}% | دقة التحقق: {val_acc:.2f}%")
    
                if loss is not None:
                   results.append(f"   → فقدان النموذج: {loss:.4f}")
                   results.append(f"   → الدقة: {accuracy*100:.2f}%")
                else:
                   results.append("   → تقييم النموذج: غير متاح (لا توجد بيانات اختبار كافية)")
    
                results.append(
                    f"   → التوقع القصير: {'صعود' if lstm_pred==1 else 'هبوط'}"
                )
                results.append(
                    f"   → الاتجاه المتوسط (10 شمعات): {medium_trend}"
                )

            # ====== الإصلاح الرئيسي: حساب الاتجاه النهائي ======
            # تحديد الاتجاه بناءً على توافق النماذج الثلاثة
            model_predictions = {
                "xgb": xgb_pred_label,
                "onnx": onnx_pred_label,
                "lstm": "صعود" if lstm_pred == 1 else "هبوط"
            }
            
            # حساب عدد التوقعات الصعودية والهبوطية
            bull_count = sum(1 for pred in model_predictions.values() if pred == "صعود")
            bear_count = sum(1 for pred in model_predictions.values() if pred == "هبوط")
            
            # تحديد الاتجاه النهائي
            if bull_count >= 2:  # إذا كان هناك توافق صعودي (غالبية)
                final_label = "صعود"
                confidence = max(0.5, (xgb_acc/100 + onnx_acc/100 + lstm_acc/100) / 3)
            elif bear_count >= 2:  # إذا كان هناك توافق هبوطي (غالبية)
                final_label = "هبوط"
                confidence = max(0.5, (xgb_acc/100 + onnx_acc/100 + lstm_acc/100) / 3)
            else:  # في حالة التعادل أو عدم التوافق
                # الأفضلية للاتجاه المتوسط (LSTM)
                final_label = medium_trend
                confidence = lstm_acc / 100 if lstm_acc > 0 else 0.6
            
            # احتساب TP و SL
            volatility = (df["high"].max() - df["low"].min()) / df["close"].mean()
            base_tp_ratios = get_base_tp_ratios(time_frame)
            
            direction_english = "up" if final_label == "صعود" else "down"
            tp_targets, stop_loss = smart_targets_sl(
                price, direction_english, confidence, volatility, base_tp_ratios
            )
            
            # حساب النسب المئوية الصحيحة
            tp_percent = []
            for tp in tp_targets:
                if final_label == "صعود":
                    pct = (tp - price) / price * 100
                else:
                    pct = (price - tp) / price * 100
                tp_percent.append(round(pct, 2))

            # إضافة النتائج
            predictions.append(
                {
                    "time_frame": time_frame,
                    "xgb_acc": xgb_acc,
                    "xgb_pred": xgb_pred_label,
                    "onnx_acc": onnx_acc,
                    "onnx_pred": onnx_pred_label,
                    "lstm_acc": lstm_acc,
                    "lstm_pred": "صعود" if lstm_pred == 1 else "هبوط",
                    "medium_trend": medium_trend,
                    "final_direction": final_label,
                    "tp_targets": tp_targets,
                    "tp_percent": tp_percent,
                    "sl": stop_loss,
                    "sl_pct": round(abs((price - stop_loss) / price) * 100, 2),
                    "price": price,
                    "confidence": confidence,
                }
            )
            
            # تنظيف الذاكرة
            del session
            gc.collect()
            clear_keras_memory()

        except Exception as e:
            logger.error(f"Error in time_frame processing {time_frame}: {traceback.format_exc()}")
            results.append(f"❌ خطأ في معالجة الفريم {time_frame}: {str(e)}")

# في مكان مناسب بعد الحصول على predictions
    enhanced_result = enhance_trading_signals(
         symbol=symbol,
    exchange=exchange,
    predictions=predictions,
    df=df,  # بيانات الإطار الزمني الحالي
    current_accuracy=acc,  # دقة النموذج الحالية
    time_frame=time_frame,
    base_position_size=0.1  # حجم الصفقة الأساسي
)

# استخدام النتيجة المحسنة
    final_label = enhanced_result['enhanced_signal']
    position_size = enhanced_result['position_size']

# إضافة المعلومات إلى النتائج
    results.append(f"الإشارة المحسنة: {final_label}")
    results.append(f"حجم الصفقة المقترح: {position_size}")

# ====== تعريف متغيرات افتراضية لتفادي الأخطاء ======
    overall_direction = "غير محدد"
    tp1_global = tp2_global = tp3_global = 0
    global_sl = 0
    avg_confidence = 0


    if predictions:

        # حساب متوسط الثقة بشكل صحيح
        confidence_values = [p.get('confidence', 0) for p in predictions if 'confidence' in p]
        if confidence_values:
           avg_confidence = sum(confidence_values) / len(confidence_values)
        else:
           avg_confidence = 0.5  # قيمة افتراضية إذا لم توجد ثقة
    
        # حفظ بيانات الإشارة للاستخدام في التداول التلقائي
        signal_data = {
            'symbol': symbol,
            'predictions': predictions,
            'overall_direction': overall_direction,
            'global_tp': [tp1_global, tp2_global, tp3_global],
            'global_sl': global_sl,
            'avg_confidence': avg_confidence
        }        
       # محاولة فتح صفقة تلقائية (سيتم تنفيذها في الخلفية)
        if auto_trade_manager.auto_trading_enabled:
            # نستخدم asyncio.create_task لتنفيذ الأمر في الخلفية
            asyncio.create_task(auto_trade_manager.check_and_open_trade(symbol, signal_data))
        
       
    # ====== بناء رسالة التحليل التراكمي ======
    if not predictions:
        error_msg = "❌ فشل في توليد أي توقعات. الأسباب المحتملة:\n1. عدم توفر بيانات كافية من البورصة\n2. مشاكل في اتصال الإنترنت\n3. الزوج غير مدعوم أو رمز غير صحيح\n4. قيم مفقودة في البيانات بعد المعالجة"
        send_telegram_message(error_msg)
        return [error_msg], None, None

    # خريطة الدقائق التراكمية لكل إطار زمني
    cumulative_minutes_map = {
        '1m': 1,
        '3m': 4,    # 1+3
        '5m': 9,    # 4+5
        '15m': 24,  # 9+15
        '30m': 54   # 24+30
    }
    
    # تجميع التنبؤات التراكمية
    cumulative_predictions = []
    for timeframe in ['1m', '3m', '5m', '15m', '30m']:
        pred = next((p for p in predictions if p['time_frame'] == timeframe), None)
        if pred:
            cumulative_minutes = cumulative_minutes_map.get(timeframe, 0)
            cumulative_predictions.append({
                'timeframe': timeframe,
                'cumulative_minutes': cumulative_minutes,
                'direction': pred['final_direction'],
                'price': pred['price'],
                'tp_targets': pred['tp_targets'],
                'tp_percent': pred['tp_percent'],
                'sl': pred['sl'],
                'sl_pct': pred['sl_pct'],
                'confidence': pred['confidence'],
                'xgb_pred': pred['xgb_pred'],
                'xgb_acc': pred['xgb_acc'],
                'onnx_pred': pred['onnx_pred'],
                'onnx_acc': pred['onnx_acc'],
                'lstm_pred': pred['lstm_pred'],
                'lstm_acc': pred['lstm_acc'],
            })

    # بناء الرسالة التراكمية
    message_lines = []
    if cumulative_predictions:
        current_price = cumulative_predictions[0]['price']  # نستخدم سعر الفريم الأول
    else:
        current_price = get_real_time_price(symbol) or 0

    message_lines.append(f"📊 **تحليل تراكمي متعدد الفريمات - {symbol}**")
    message_lines.append(f"🔵 السعر الحالي: **{current_price:.6f}**\n")
    message_lines.append("🧠 **تفاصيل النماذج لكل إطار زمني:**")

    # أسماء الفريمات بالعربية
    timeframe_names = {
        '1m': 'دقيقة واحدة',
        '3m': '3 دقائق',
        '5m': '5 دقائق',
        '15m': '15 دقيقة',
        '30m': '30 دقيقة'
    }

    # تفاصيل النماذج لكل إطار زمني
    for pred in cumulative_predictions:
        tf_name = timeframe_names.get(pred['timeframe'], pred['timeframe'])
        
        # تنسيق الأهداف كأسعار ونسب مئوية
        tp_str = ""
        for i, (tp_price, tp_pct) in enumerate(zip(pred['tp_targets'], pred['tp_percent'])):
            tp_str += f"    - TP{i+1}: **{tp_price:.6f}** ({tp_pct:+.2f}%)\n"
        
        message_lines.append(
            f"\n⏱ **{tf_name} (الوقت التراكمي: {pred['cumulative_minutes']} دقيقة):**\n"
            f"  📊 الإتجاه النهائي: **{'صعود 🔺' if pred['direction'] == 'صعود' else 'هبوط 🔻'}** (ثقة: {pred['confidence']*100:.1f}%)\n"
            f"  🤖 XGBoost: {pred['xgb_pred']} (دقة: {pred['xgb_acc']:.1f}%)\n"
            f"  ⚡ ONNX: {pred['onnx_pred']} (ثقة: {pred['onnx_acc']:.1f}%)\n"
            f"  🧠 LSTM: {pred['lstm_pred']} (دقة: {pred['lstm_acc']:.1f}%)\n"
            f"🎯 **الأهداف:**\n"
            f"{tp_str}"
            f"  🛑 وقف الخسارة: **{pred['sl']:.6f}** ({pred['sl_pct']:.2f}%)"
        )

    # ====== التحليل النهائي (جمع كل الفريمات) ======
    if cumulative_predictions:
        # تحديد الاتجاه العام
        bull_count = sum(1 for p in cumulative_predictions if p['direction'] == 'صعود')
        bear_count = len(cumulative_predictions) - bull_count
        overall_direction = "صعود" if bull_count > bear_count else "هبوط"
        
        # جمع جميع أهداف TP من كل الفريمات
        all_tp1 = []
        all_tp2 = []
        all_tp3 = []
        for pred in cumulative_predictions:
            if len(pred['tp_targets']) >= 3:
                all_tp1.append(pred['tp_targets'][0])
                all_tp2.append(pred['tp_targets'][1])
                all_tp3.append(pred['tp_targets'][2])
        
        if overall_direction == "صعود":
            tp1_global = np.mean(all_tp1) if all_tp1 else 0
            tp2_global = np.max(all_tp2) if all_tp2 else 0
            tp3_global = np.max(all_tp3) if all_tp3 else 0
        else:
            tp1_global = np.mean(all_tp1) if all_tp1 else 0
            tp2_global = np.min(all_tp2) if all_tp2 else 0
            tp3_global = np.min(all_tp3) if all_tp3 else 0
        
        # حساب النسب المئوية
        if current_price:
            tp1_pct = (tp1_global - current_price) / current_price * 100
            tp2_pct = (tp2_global - current_price) / current_price * 100
            tp3_pct = (tp3_global - current_price) / current_price * 100
        else:
            tp1_pct = tp2_pct = tp3_pct = 0
        
        # حساب وقف الخسارة العالمي: متوسط وقف الخسارة من كل الفريمات
        global_sl = np.mean([p['sl'] for p in cumulative_predictions])
        global_sl_pct = abs((global_sl - current_price) / current_price) * 100
        
        # إضافة التحليل النهائي
        message_lines.append("\n🌟 **التحليل النهائي (مجمع من كل الفريمات):**")
        message_lines.append(f"📌 الإتجاه العام: **{'صعودي 🔺' if overall_direction == 'صعود' else 'هبوطي 🔻'}**")
        message_lines.append(f"🎯 الهدف الأول: **{tp1_global:.6f}** ({tp1_pct:+.2f}%)")
        message_lines.append(f"🎯🎯 الهدف الثاني: **{tp2_global:.6f}** ({tp2_pct:+.2f}%)")
        message_lines.append(f"🎯🎯🎯 الهدف الثالث: **{tp3_global:.6f}** ({tp3_pct:+.2f}%)")
        message_lines.append(f"🛑 وقف الخسارة العالمي: **{global_sl:.6f}** ({global_sl_pct:.2f}%)")
        
        # حساب متوسط الثقة
        avg_confidence = sum(p['confidence'] for p in cumulative_predictions) / len(cumulative_predictions) * 100
        message_lines.append(f"🔐 متوسط الثقة: **{avg_confidence:.1f}%**")
    else:
        message_lines.append("\n⚠️ لم يتم إنشاء توقعات تراكمية")


        # في الجزء الرئيسي من get_signal، بعد الحصول على predictions
    if predictions:
        # تحديد مستويات الدعم والمقاومة من التوقعات
        current_price = predictions[0]['price'] if predictions else 0
        key_levels = identify_key_levels(predictions, current_price)

        # حساب متوسط الثقة بشكل صحيح
        confidence_values = [p.get('confidence', 0) for p in predictions if 'confidence' in p]
        if confidence_values:
            avg_confidence = sum(confidence_values) / len(confidence_values)
        else:
            avg_confidence = 0.5

        # تحديد الاتجاه العام (قبل استخدامه)
        bull_count = sum(1 for p in predictions if p['final_direction'] == 'صعود')
        overall_direction = "صعود" if bull_count > len(predictions) / 2 else "هبوط"

        # حساب وقف الخسارة العالمي باستخدام المستويات
        if key_levels:
            # حساب وقف الخسارة بناءً على الاتجاه
            if overall_direction == "صعود":
                global_sl = calculate_smart_stop_loss("BUY", current_price, key_levels)
            else:
                global_sl = calculate_smart_stop_loss("SELL", current_price, key_levels)
        else:
            # استخدام الطريقة القديمة كاحتياطي
            global_sl = current_price * (0.98 if overall_direction == "صعود" else 1.02)

        # حفظ بيانات الإشارة للاستخدام في التداول التلقائي
        signal_data = {
            'symbol': symbol,
            'predictions': predictions,
            'overall_direction': overall_direction,
            'global_tp': [tp1_global, tp2_global, tp3_global],
            'global_sl': global_sl,
            'avg_confidence': avg_confidence,
            'analysis': {
                'key_levels': key_levels  # إضافة المستويات للتحليل
            }
        }

    # ====== توليد شموع مستقبلية ======
    # ====== AI-Powered Future Candle Generation ======
    image_path = None
    try:
        if predictions and not df.empty:
            # 1. Collect and resample data from all timeframes
            all_timeframes = []
            for timeframe in ["1m", "3m", "5m", "15m", "30m"]:  
                if timeframe in timeframe_data and not timeframe_data[timeframe].empty:
                    # Ensure datetime index
                    tf_data = timeframe_data[timeframe].copy()
                    tf_data.index = pd.to_datetime(tf_data.index)
                    
                    # Clean data - remove any NaN in OHLC
                    tf_data.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
                    
                    # Resample to 5-minute timeframe
                    resampled = tf_data.resample('5T').agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum'
                    })
                    # Forward fill any missing values
                    resampled.ffill(inplace=True)
                    all_timeframes.append(resampled)
            
            # Combine all resampled data
            combined_5m = pd.concat(all_timeframes).sort_index()
            combined_5m = combined_5m[~combined_5m.index.duplicated(keep='last')]
            combined_5m = combined_5m.sort_index()
            
            # Clean combined data - remove any NaN in OHLC
            combined_5m.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
            
            if combined_5m.empty:
                raise ValueError("No valid data available for 5-minute timeframe")
            
            # 2. Enhanced AI consensus with all timeframes
            ai_predictions = []
            for p in predictions:
                timeframe_weights = {  
                    "1m": 0.15,
                    "3m": 0.2,
                    "5m": 0.3,
                    "15m": 0.2,
                    "30m": 0.15
                }
                weight = timeframe_weights.get(p['time_frame'], 0.1)
                
                weighted_pred = weight * (
                    0.4 * (1 if p['final_direction'] == 'صعود' else 0) +
                    0.3 * (1 if p['xgb_pred'] == 'صعود' else 0) +
                    0.3 * (1 if p['lstm_pred'] == 'صعود' else 0)
                )
                
                ai_predictions.append({
                    'time_frame': p['time_frame'],
                    'weighted_pred': weighted_pred,
                    'confidence': p['confidence']
                })
            
            # 3. Calculate trend strength
            trend_strength = min(1.0, max(0.1, sum(
                p['weighted_pred'] * p['confidence'] for p in ai_predictions
            ) / sum(timeframe_weights.values())))
            
            # 4. Calculate volatility
            volatility = combined_5m['close'].pct_change().std() * 100
            if np.isnan(volatility) or volatility < 0.5:
                volatility = 0.5 + (trend_strength * 0.3)
            
            # 5. استخدام الدالة المحسنة لتوليد الشموع المستقبلية
            last_close = combined_5m['close'].iloc[-1]
            future_df = generate_more_natural_future_candles(
                last_close=last_close,
                direction="up" if overall_direction == "صعود" else "down",
                num=150,
                base_volatility=volatility/100,  # تحويل النسبة المئوية إلى قيمة عشرية
                trend_strength=trend_strength,
                recent_data=combined_5m if not combined_5m.empty else None,
                targets=[tp1_global, tp2_global, tp3_global]
            )
            
            # 6. Convert to proper DatetimeIndex
            last_time = pd.Timestamp.utcnow()
            future_times = pd.DatetimeIndex([last_time + pd.Timedelta(minutes=5*(i+1)) for i in range(len(future_df))])
            future_df.index = future_times
            
            # 7. Ensure index is timezone-aware
            if future_df.index.tz is None:
                future_df.index = future_df.index.tz_localize('UTC')
            
            # 8. إضافة مؤشرات فنية للفترة الطويلة
            future_df['MA20'] = future_df['close'].rolling(20, min_periods=1).mean()
            future_df['MA50'] = future_df['close'].rolling(50, min_periods=1).mean()
            future_df['EMA100'] = future_df['close'].ewm(span=100, adjust=False, min_periods=1).mean()
            
            # Bollinger Bands
            future_df['BB_middle'] = future_df['close'].rolling(20, min_periods=1).mean()
            future_df['BB_upper'] = future_df['BB_middle'] + 2 * future_df['close'].rolling(20, min_periods=1).std()
            future_df['BB_lower'] = future_df['BB_middle'] - 2 * future_df['close'].rolling(20, min_periods=1).std()

            # ====== Professional Dark Theme Chart ======
            DARK_BG = '#121723'
            GRID_COLOR = '#2a3b5a'
            TEXT_COLOR = '#e0e6f0'
            UP_COLOR = '#00c853'
            DOWN_COLOR = '#ff5252'
            
            mc = mpf.make_marketcolors(
                up=UP_COLOR,
                down=DOWN_COLOR,
                wick={'up': UP_COLOR, 'down': DOWN_COLOR},
                edge={'up': UP_COLOR, 'down': DOWN_COLOR},
                volume='#4d648d',
                ohlc='i'
            )
            
            s = mpf.make_mpf_style(
                base_mpl_style='dark_background',
                marketcolors=mc,
                gridstyle="--",
                gridcolor=GRID_COLOR,
                facecolor=DARK_BG,
                edgecolor=TEXT_COLOR,
                figcolor=DARK_BG,
                rc={
                    'font.size': 10,
                    'axes.labelcolor': TEXT_COLOR,
                    'text.color': TEXT_COLOR,
                    'xtick.color': TEXT_COLOR,
                    'ytick.color': TEXT_COLOR,
                    'axes.titlecolor': TEXT_COLOR,
                    'axes.edgecolor': GRID_COLOR,
                    'axes.labelweight': 'bold'
                }
            )
            
            # Prepare addplots للمزيد من المؤشرات
            apds = [
                mpf.make_addplot(future_df['MA20'], color='#3498db', width=1, panel=0),
                mpf.make_addplot(future_df['MA50'], color='#9b59b6', width=1.5, panel=0),
                mpf.make_addplot(future_df['EMA100'], color='#e67e22', width=1.5, panel=0),
                mpf.make_addplot(future_df['BB_upper'], color='#95a5a6', width=0.8, linestyle='--', panel=0),
                mpf.make_addplot(future_df['BB_lower'], color='#95a5a6', width=0.8, linestyle='--', panel=0),
                mpf.make_addplot(
                    pd.Series([tp1_global]*len(future_df), index=future_df.index),
                    type='line', color='#2ecc71', width=1.5, linestyle='--', panel=0),
                mpf.make_addplot(
                    pd.Series([tp2_global]*len(future_df), index=future_df.index),
                    type='line', color='#3498db', width=2, linestyle='-.', panel=0),
                mpf.make_addplot(
                    pd.Series([tp3_global]*len(future_df), index=future_df.index),
                    type='line', color='#9b59b6', width=2.5, panel=0),
                mpf.make_addplot(
                    pd.Series([last_close]*len(future_df), index=future_df.index),
                    type='line', color='#e74c3c', width=1.5, linestyle=':', panel=0),
                mpf.make_addplot(future_df['volume'], type='bar', panel=1, color='#4d648d', ylabel='Volume', alpha=0.7)
            ]
            
            # Create figure with corrected index
            fig, axes = mpf.plot(
                future_df[['open', 'high', 'low', 'close']],
                type="candle",
                style=s,
                volume=False,
                addplot=apds,
                figratio=(20, 12),  # زيادة حجم الرسم للشموع الكثيرة
                panel_ratios=(6, 1),
                returnfig=True,
                datetime_format='%H:%M',
                xrotation=45,
                tight_layout=True,
                scale_padding={'left': 0.1, 'top': 0.95, 'right': 0.95, 'bottom': 0.2}
            )
            
            # Setup titles and labels
            ax_main = axes[0]
            ax_volume = axes[2]
            
            # Dynamic price formatting based on symbol
            if last_close > 1000:
                decimals = 2
            elif last_close > 10:
                decimals = 4
            else:
                decimals = 6
            
            # Format price with proper decimals
            ax_main.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"{x:,.{decimals}f}"))
            
            # تحديث العنوان ليعكس عدد الشموع الجديد
            ax_main.set_title(
                f"{symbol} - AI Price Prediction (Next {len(future_df)*5} Minutes / {len(future_df)} Candles)\n"
                f"Overall Trend: {'BULLISH' if overall_direction == 'صعود' else 'BEARISH'} | "
                f"Confidence: {avg_confidence:.1f}% | Volatility: {volatility:.2f}%",
                fontsize=14,
                fontweight='bold',
                pad=20
            )
            
            ax_main.set_ylabel("Price", fontsize=11, fontweight='bold')
            ax_volume.set_ylabel("Volume", fontsize=11, fontweight='bold')
            
            # إضافة وسيلة إيضاح للمؤشرات
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='#3498db', lw=2, label='MA20'),
                Line2D([0], [0], color='#9b59b6', lw=2, label='MA50'),
                Line2D([0], [0], color='#e67e22', lw=2, label='EMA100'),
                Line2D([0], [0], color='#2ecc71', lw=2, linestyle='--', label='Target 1'),
                Line2D([0], [0], color='#3498db', lw=2, linestyle='-.', label='Target 2'),
                Line2D([0], [0], color='#9b59b6', lw=2, label='Target 3'),
                Line2D([0], [0], color='#e74c3c', lw=2, linestyle=':', label='Entry Price')
            ]
            
            ax_main.legend(handles=legend_elements, loc='upper left', fontsize=8, 
                          facecolor=DARK_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
            
            # Save high-quality image
            images_dir = os.path.join(model_dir, "ai_charts")
            os.makedirs(images_dir, exist_ok=True)
            image_filename = f"{symbol.replace('/', '_')}_150_candles_pred_{int(time.time())}.png"
            image_path = os.path.join(images_dir, image_filename)
            
            fig.savefig(
                image_path, 
                bbox_inches="tight", 
                dpi=150,  # تقليل الدقة قليلاً للحفاظ على حجم الملف
                pad_inches=0.5,
                facecolor=DARK_BG
            )
            plt.close(fig)
            
            results.append(f"✅ AI multi-timeframe prediction chart with 150 candles saved: {image_path}")
            
    except Exception as e:
        logger.error(f"Error in chart generation: {traceback.format_exc()}")
        send_telegram_message(f"⚠️ Chart generation error: {str(e)[:200]}")
        image_path = None
    # ====== تجهيز الرسالة النهائية ======
    def split_message(text, max_length=4000):
        lines = text.split("\n")
        parts = []
        current = ""
        for line in lines:
            if len(current) + len(line) + 1 <= max_length:
                current += line + "\n"
            else:
                parts.append(current.strip())
                current = line + "\n"
        if current:
            parts.append(current.strip())
        return parts

    final_text = "\n".join(message_lines)
    message_parts = split_message(final_text)

    
    def clear_memory():
        tf.keras.backend.clear_session()
        gc.collect()


    # تنظيف الذاكرة
    gc.collect()
    clear_keras_memory()

    if image_path and os.path.exists(image_path):
        return message_parts, image_path, signal_data
    else:
        return message_parts, None, signal_data
    

def calculate_weighted_consensus(predictions):
    """
    نظام تصويت مرجح يعتمد على أداء كل نموذج
    """
    # أوزان النماذج بناءً على دقتها التاريخية
    model_weights = {
        "xgb": predictions["xgb_acc"] / 100,
        "onnx": predictions["onnx_acc"] / 100,
        "lstm": predictions["lstm_acc"] / 100
    }
    
    # حساب التصويت المرجح
    weighted_bull = 0
    weighted_bear = 0
    
    if predictions["xgb_pred"] == "صعود":
        weighted_bull += model_weights["xgb"]
    else:
        weighted_bear += model_weights["xgb"]
        
    if predictions["onnx_pred"] == "صعود":
        weighted_bull += model_weights["onnx"]
    else:
        weighted_bear += model_weights["onnx"]
        
    if predictions["lstm_pred"] == "صعود":
        weighted_bull += model_weights["lstm"]
    else:
        weighted_bear += model_weights["lstm"]
    
    # تحديد الاتجاه بناءً على الترجيح
    if weighted_bull > weighted_bear:
        return "صعود", weighted_bull / (weighted_bull + weighted_bear)
    elif weighted_bear > weighted_bull:
        return "هبوط", weighted_bear / (weighted_bull + weighted_bear)
    else:
        # التعادل - نعود للاتجاه طويل المدى
        return predictions["medium_trend"], 0.5

def get_market_trend(symbol, exchange):
    """
    تحليل الاتجاه العام للسوق باستخدام أطر زمنية أعلى
    """
    try:
        # جلب بيانات الإطار اليومي
        daily_data = exchange.fetch_ohlcv(symbol, '1d', limit=30)
        df_daily = pd.DataFrame(daily_data, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
        
        # حساب المتوسطات المتحركة
        df_daily['sma_20'] = df_daily['close'].rolling(20).mean()
        df_daily['sma_50'] = df_daily['close'].rolling(50).mean()
        
        # تحديد الاتجاه
        last_close = df_daily['close'].iloc[-1]
        sma_20 = df_daily['sma_20'].iloc[-1]
        sma_50 = df_daily['sma_50'].iloc[-1]
        
        if last_close > sma_20 > sma_50:
            return "صعود", 0.7  # اتجاه صعودي قوي
        elif last_close < sma_20 < sma_50:
            return "هبوط", 0.7  # اتجاه هبوطي قوي
        else:
            return "محايد", 0.5  # سوق متذبذب
            
    except Exception as e:
        logging.error(f"Error getting market trend: {str(e)}")
        return "غير محدد", 0.5

def analyze_volatility_impact(df, current_volatility):
    """
    تحليل تأثير التقلبات على دقة النماذج
    """
    # افتراض أن النماذج الأبسط (XGBoost) أفضل في الأسواق عالية التقلب
    # بينما النماذج المعقدة (LSTM) أفضل في الأسواق المستقرة
    
    volatility_level = "عالي" if current_volatility > 0.02 else "منخفض"
    
    if volatility_level == "عالي":
        return {"xgb": 1.2, "onnx": 1.0, "lstm": 0.8}  # زيادة وزن XGBoost
    else:
        return {"xgb": 0.9, "onnx": 1.0, "lstm": 1.1}  # زيادة وزن LSTM

def expert_conflict_resolution(predictions, market_trend, volatility_impact):
    """
    نظام خبير لحل التعارض بين النماذج
    """
    conflict_type = ""
    
    # تحديد نوع التعارض
    if predictions["xgb_pred"] != predictions["lstm_pred"]:
        conflict_type = "تقليدي vs ذكاء اصطناعي"
    elif predictions["xgb_pred"] != predictions["onnx_pred"]:
        conflict_type = "تقليدي vs محسن"
    else:
        conflict_type = "ذكاء اصطناعي vs محسن"
    
    # قواعد القرار based على نوع التعارض واتجاه السوق
    if conflict_type == "تقليدي vs ذكاء اصطناعي":
        if market_trend in ["صعود", "هبوط"]:
            # في وجود اتجاه واضح، نرجح النموذج التقليدي
            return predictions["xgb_pred"], f"اتباع الاتجاه العام ({market_trend})"
        else:
            # في السوق المتذبذب، نرجح الذكاء الاصطناعي
            return predictions["lstm_pred"], "سوق متذبذب - تفضيل LSTM"
    
    # ... قواعد أخرى لأنواع التعارض المختلفة
    
    return predictions["xgb_pred"], "قاعدة افتراضية"

def generate_intelligent_prediction_report(signal_data, symbol):
    """
    توليد تقرير ذكي مبسط لتحليل توقعات الذكاء الاصطناعي مع إرسال تلقائي
    """
    if not signal_data or 'predictions' not in signal_data:
        error_msg = "⚠️ لا توجد بيانات تنبؤية كافية لتحليلها"
        # إرسال رسالة الخطأ إلى Telegram
        asyncio.create_task(send_telegram_message_async(error_msg))
        return error_msg
    
    predictions = signal_data['predictions']
    overall_direction = signal_data.get('overall_direction', 'غير محدد')
    global_tp = signal_data.get('global_tp', [])
    global_sl = signal_data.get('global_sl', 0)
    avg_confidence = signal_data.get('avg_confidence', 0)
    
    # الحصول على السعر الحالي
    current_price = predictions[0]['price'] if predictions else 0
    
    # تحليل التوقعات من جميع الأطر الزمنية
    analysis = analyze_multiple_timeframes(predictions)
    
    # إنشاء التقرير الذكي
    report = generate_smart_report(analysis, symbol, current_price, 
                                 overall_direction, global_tp, global_sl, avg_confidence)
    
    # إرسال التقرير تلقائياً إلى Telegram
    asyncio.create_task(send_prediction_report_to_telegram(report, symbol))
    
    return report

async def send_prediction_report_to_telegram(report, symbol):
    """
    إرسال تقرير التنبؤات إلى Telegram تلقائياً
    """
    try:
        # تقسيم التقرير الطويل إلى أجزاء إذا لزم الأمر
        if len(report) > 4000:
            parts = []
            current_part = ""
            
            lines = report.split('\n')
            for line in lines:
                if len(current_part) + len(line) + 1 > 4000:
                    parts.append(current_part)
                    current_part = line + '\n'
                else:
                    current_part += line + '\n'
            
            if current_part:
                parts.append(current_part)
            
            # إرسال الأجزاء مع تأخير بينها
            for i, part in enumerate(parts):
                await send_telegram_message_async(part)
                if i < len(parts) - 1:
                    await asyncio.sleep(1)
        else:
            await send_telegram_message_async(report)
            
        logging.info(f"✅ تم إرسال التقرير الذكي لـ {symbol} إلى Telegram")
        
    except Exception as e:
        logging.error(f"❌ فشل إرسال التقرير إلى Telegram: {str(e)}")

def analyze_multiple_timeframes(predictions):
    """
    تحليل متعمق للتنبؤات من جميع الأطر الزمنية
    """
    analysis = {
        'timeframe_analysis': {},
        'consensus_strength': 0,
        'volatility_estimate': 0,
        'time_horizons': {},
        'key_levels': {}
    }
    
    if not predictions:
        return analysis
    
    # تحليل كل إطار زمني
    for pred in predictions:
        tf = pred['time_frame']
        analysis['timeframe_analysis'][tf] = {
            'direction': pred['final_direction'],
            'confidence': pred['confidence'],
            'models_agreement': calculate_models_agreement(pred),
            'time_horizon': get_time_horizon_minutes(tf)
        }
    
    # حساب قوة الإجماع
    bull_count = sum(1 for p in predictions if p['final_direction'] == 'صعود')
    analysis['consensus_strength'] = bull_count / len(predictions) if predictions else 0.5
    
    # تقدير التقلبات
    analysis['volatility_estimate'] = estimate_volatility(predictions)
    
    # تحديد المستويات الرئيسية
    analysis['key_levels'] = identify_key_levels(predictions)
    
    # تحليل الآفاق الزمنية
    analysis['time_horizons'] = analyze_time_horizons(predictions)
    
    return analysis

def calculate_models_agreement(prediction):
    """
    حساب درجة توافق النماذج المختلفة
    """
    models = ['xgb_pred', 'onnx_pred', 'lstm_pred']
    same_direction_count = sum(1 for model in models 
                              if prediction.get(model, '') == prediction['final_direction'])
    return same_direction_count / len(models)

def get_time_horizon_minutes(timeframe):
    """
    تحويل الإطار الزمني إلى دقائق للتنبؤ
    """
    horizons = {
        '1m': 30,    # 30 دقيقة توقع
        '3m': 90,    # 1.5 ساعة توقع
        '5m': 150,   # 2.5 ساعة توقع
        '15m': 300,  # 5 ساعات توقع
        '30m': 600   # 10 ساعات توقع
    }
    return horizons.get(timeframe, 60)

def estimate_volatility(predictions):
    """
    تقدير مستوى التقلبات المتوقعة
    """
    if not predictions:
        return 0.5  # متوسط
    
    volatility_scores = []
    for pred in predictions:
        # حساب التقلب بناء على المدى بين السعر والأهداف
        price = pred['price']
        max_tp = max(pred['tp_targets']) if pred['tp_targets'] else price
        min_tp = min(pred['tp_targets']) if pred['tp_targets'] else price
        sl = pred['sl']
        
        # المدى النسبي
        upper_range = abs(max_tp - price) / price
        lower_range = abs(price - min(sl, min_tp)) / price
        total_range = upper_range + lower_range
        
        volatility_scores.append(total_range)
    
    return sum(volatility_scores) / len(volatility_scores)

def identify_key_levels(predictions, current_price=None):
    """
    تحديد المستويات السعرية الرئيسية مع مراعاة السعر الحالي
    """
    resistance_levels = []
    support_levels = []
    
    for pred in predictions:
        if pred['final_direction'] == 'صعود':
            # في الصعود، الأهداف تمثل مقاومة والمحافظة على الدعم
            resistance_levels.extend(pred['tp_targets'])
            support_levels.append(pred['sl'])
        else:
            # في الهبوط، الأهداف تمثل دعم والمحافظة على المقاومة
            support_levels.extend(pred['tp_targets'])
            resistance_levels.append(pred['sl'])
    
    # تصفية المستويات بناءً على السعر الحالي إذا كان متاحاً
    if current_price is not None:
        # للدعم: نريد المستويات تحت السعر الحالي
        support_levels = [s for s in support_levels if s < current_price]
        # للمقاومة: نريد المستويات فوق السعر الحالي
        resistance_levels = [r for r in resistance_levels if r > current_price]
    
    # تجميع وتصنيف المستويات مع مراعاة القوة
    key_levels = {
        'strong_resistance': sorted(set(resistance_levels), reverse=True)[:3],
        'strong_support': sorted(set(support_levels))[:3],
        'primary_targets': [pred['tp_targets'][0] for pred in predictions if pred['tp_targets']],
        'secondary_targets': [pred['tp_targets'][1] for pred in predictions if len(pred['tp_targets']) > 1]
    }
    
    return key_levels

def analyze_time_horizons(predictions):
    """
    تحليل الآفاق الزمنية المختلفة
    """
    time_horizons = {
        'short_term': {'direction': None, 'confidence': 0, 'targets': []},
        'medium_term': {'direction': None, 'confidence': 0, 'targets': []},
        'long_term': {'direction': None, 'confidence': 0, 'targets': []}
    }
    
    for pred in predictions:
        tf = pred['time_frame']
        horizon_minutes = get_time_horizon_minutes(tf)
        
        if horizon_minutes <= 120:  # قصير المدى (ساعتان)
            update_horizon_analysis(time_horizons['short_term'], pred)
        elif horizon_minutes <= 360:  # متوسط المدى (6 ساعات)
            update_horizon_analysis(time_horizons['medium_term'], pred)
        else:  # طويل المدى
            update_horizon_analysis(time_horizons['long_term'], pred)
    
    return time_horizons

def update_horizon_analysis(horizon, prediction):
    """
    تحديث تحليل الأفق الزمني
    """
    if horizon['direction'] is None:
        horizon['direction'] = prediction['final_direction']
        horizon['confidence'] = prediction['confidence']
        horizon['targets'] = prediction['tp_targets']
    else:
        # دمج التوقعات
        if prediction['final_direction'] == horizon['direction']:
            horizon['confidence'] = (horizon['confidence'] + prediction['confidence']) / 2
        else:
            # في حالة التعارض، نأخذ بالأعلى ثقة
            if prediction['confidence'] > horizon['confidence']:
                horizon['direction'] = prediction['final_direction']
                horizon['confidence'] = prediction['confidence']
                horizon['targets'] = prediction['tp_targets']

def generate_smart_report(analysis, symbol, current_price, overall_direction, global_tp, global_sl, avg_confidence):
    """
    توليد تقرير ذكي وسهل الفهم
    """
    report_lines = []
    
    # العنوان الرئيسي
    report_lines.append(f"🧠 <b>التقرير الذكي لتحليل {symbol}</b>")
    report_lines.append("=" * 50)
    
    # الملخص التنفيذي
    report_lines.append("\n🎯 <b>الملخص التنفيذي:</b>")
    
    direction_emoji = "🔺" if overall_direction == "صعود" else "🔻"
    report_lines.append(f"• الاتجاه العام: <b>{overall_direction}</b> {direction_emoji}")
    report_lines.append(f"• الثقة الإجمالية: <b>{avg_confidence:.1f}%</b>")
    
    # تحليل التقلبات
    volatility_desc = get_volatility_description(analysis['volatility_estimate'])
    report_lines.append(f"• مستوى التقلبات المتوقع: <b>{volatility_desc}</b>")
    
    # قوة الإجماع
    consensus_strength = analysis['consensus_strength'] * 100
    report_lines.append(f"• قوة إجماع النماذج: <b>{consensus_strength:.1f}%</b>")
    
    # التنبؤ الزمني
    report_lines.append("\n⏰ <b>التنبؤ الزمني:</b>")
    
    # المدى القصير (1-2 ساعة)
    short_term = analysis['time_horizons']['short_term']
    if short_term['direction']:
        report_lines.append(f"• خلال الساعة القادمة: اتجاه <b>{short_term['direction']}</b>")
        if short_term['targets']:
            target = short_term['targets'][0]
            change_pct = ((target - current_price) / current_price) * 100
            report_lines.append(f"  → مستهدف أول: {target:.6f} ({change_pct:+.2f}%)")
    
    # المدى المتوسط (2-6 ساعات)
    medium_term = analysis['time_horizons']['medium_term']
    if medium_term['direction']:
        report_lines.append(f"• خلال 2-6 ساعات: اتجاه <b>{medium_term['direction']}</b>")
        if medium_term['targets'] and len(medium_term['targets']) > 1:
            target = medium_term['targets'][1]
            change_pct = ((target - current_price) / current_price) * 100
            report_lines.append(f"  → مستهدف ثاني: {target:.6f} ({change_pct:+.2f}%)")
    
    # نطاق التداول المتوقع
    report_lines.append("\n📈 <b>نطاق التداول المتوقع:</b>")
    
    if analysis['key_levels']['strong_support'] and analysis['key_levels']['strong_resistance']:
        support = min(analysis['key_levels']['strong_support'])
        resistance = max(analysis['key_levels']['strong_resistance'])
        
        report_lines.append(f"• الدعم القوي: <b>{support:.6f}</b>")
        report_lines.append(f"• المقاومة القوية: <b>{resistance:.6f}</b>")
        report_lines.append(f"• نطاق التداول: <b>{resistance - support:.6f}</b>")
    
    # التوصيات الذكية
    report_lines.append("\n💡 <b>التوصيات الذكية:</b>")
    
    if overall_direction == "صعود":
        if avg_confidence > 70:
            report_lines.append("• 🟢 <b>فرصة شراء قوية</b> مع وضع وقف الخسارة تحت الدعم")
            report_lines.append("• 🎯 مستويات الجني: " + " | ".join(f"{tp:.6f}" for tp in global_tp[:2]))
        elif avg_confidence > 50:
            report_lines.append("• 🟡 <b>فرصة شراء متوسطة</b> - انتظر التأكيد الإضافي")
        else:
            report_lines.append("• 🔴 <b>تجنب الشراء حالياً</b> - الثقة منخفضة")
    else:
        if avg_confidence > 70:
            report_lines.append("• 🔴 <b>فرصة بيع قوية</b> مع وضع وقف الخسارة فوق المقاومة")
        elif avg_confidence > 50:
            report_lines.append("• 🟡 <b>فرصة بيع متوسطة</b> - انتظر التأكيد الإضافي")
        else:
            report_lines.append("• 🟢 <b>تجنب البيع حالياً</b> - الثقة منخفضة")
    
    # تحذيرات مهمة
    report_lines.append("\n⚠️ <b>ملاحظات مهمة:</b>")
    report_lines.append("• هذه التوقعات تعتمد على التحليل الإحصائي وليست ضمانة")
    report_lines.append("• دائماً استخدم إدارة المخاطر المناسبة")
    report_lines.append(f"• آخر تحديث: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return "\n".join(report_lines)

def get_volatility_description(volatility_score):
    """
    وصف مستوى التقلبات باللغة الطبيعية
    """
    if volatility_score < 0.02:
        return "منخفض جداً"
    elif volatility_score < 0.05:
        return "منخفض"
    elif volatility_score < 0.08:
        return "متوسط"
    elif volatility_score < 0.12:
        return "مرتفع"
    else:
        return "مرتفع جداً"

# دالة مساعدة للربط مع نظام التحليل الرئيسي
async def process_signal_and_generate_report(symbol, signal_data):
    """
    معالجة الإشارة وتوليد التقرير تلقائياً
    """
    try:
        if signal_data and 'predictions' in signal_data:
            # توليد التقرير الذكي
            report = generate_intelligent_prediction_report(signal_data, symbol)
            return report
        else:
            error_msg = f"⚠️ لا توجد بيانات كافية لإنشاء تقرير عن {symbol}"
            await send_telegram_message_async(error_msg)
            return error_msg
            
    except Exception as e:
        error_msg = f"❌ خطأ في معالجة إشارة {symbol}: {str(e)}"
        logging.error(error_msg)
        await send_telegram_message_async(error_msg)
        return error_msg


def send_telegram_message(message, max_retries=3):
    """إرسال رسالة تلجرام مع إعادة المحاولة"""
    if not TELEGRAM_TOKEN or not AUTHORIZED_USER_ID:
        raise ValueError("لم يتم تعيين متغيرات Telegram المطلوبة")
    
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": AUTHORIZED_USER_ID,
        "text": message,
        "parse_mode": "HTML"
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            print(f"تم إرسال الرسالة بنجاح: {message[:50]}...")
            return True
        except (requests.exceptions.RequestException, ConnectionResetError) as e:
            logging.warning(f"المحاولة {attempt+1}: فشل إرسال الرسالة - {str(e)}")
            time.sleep((attempt + 1) * 2)  # زيادة مهلة الانتظار تدريجياً
    
    logging.error(f"فشل إرسال الرسالة بعد {max_retries} محاولات")
    return False

# ====== تعريف دالة لإرسال الصور ======
def send_telegram_photo(photo_path, caption=""):
    global TELEGRAM_TOKEN, AUTHORIZED_USER_ID
    
    if not TELEGRAM_TOKEN or not AUTHORIZED_USER_ID:
        raise ValueError("لم يتم تعيين متغيرات Telegram المطلوبة")
    
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    
    try:
        with open(photo_path, 'rb') as photo:
            files = {'photo': photo}
            data = {'chat_id': AUTHORIZED_USER_ID}
            if caption:
                data['caption'] = caption
            
            response = requests.post(url, files=files, data=data)
            response.raise_for_status()
            print(f"تم إرسال الصورة بنجاح: {caption}")
            return True
    except Exception as e:
        print(f"فشل إرسال الصورة: {str(e)}")
        return False
# تحسين ذاكرة التخزين المؤقت
price_cache = {}
CACHE_DURATION = 300  # 5 دقائق بدلاً من 30 ثانية


async def send_telegram_message_async(message, max_retries=3, timeout=15):
    """إرسال رسالة تلجرام مع إعادة المحاولة ومعالجة الأخطاء"""
    if not TELEGRAM_TOKEN or not AUTHORIZED_USER_ID:
        print("❌ لم يتم تعيين متغيرات Telegram المطلوبة")
        return False
    
    # التحقق من أن الرسالة ليست فارغة وتنظيفها من الأحخاص الخاصة
    if not message or not message.strip():
        print("⚠️ تم تخطي إرسال رسالة فارغة")
        return False
    
    # تنظيف الرسالة من أي أحخاص قد تسبب مشاكل
    cleaned_message = message.encode('utf-8', 'ignore').decode('utf-8')
    
    # إذا كانت الرسالة طويلة جداً، تقسيمها
    if len(cleaned_message) > 4096:
        print("⚠️ الرسالة طويلة جداً، سيتم تقسيمها إلى أجزاء")
        message_parts = [cleaned_message[i:i+4000] for i in range(0, len(cleaned_message), 4000)]
        results = []
        for part in message_parts:
            success = await send_telegram_message_async(part, max_retries, timeout)
            results.append(success)
            await asyncio.sleep(1)
        return all(results)
    
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": AUTHORIZED_USER_ID,
        "text": cleaned_message,
        "parse_mode": "HTML"
    }
    
    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=timeout) as response:
                    if response.status != 200:
                        response_text = await response.text()
                        print(f"⚠️ استجابة غير ناجحة: {response.status} - {response_text}")
                        if "bad request" in response_text.lower():
                            # حاول بدون HTML إذا كان هناك مشكلة
                            payload_no_html = payload.copy()
                            payload_no_html["parse_mode"] = None
                            async with session.post(url, json=payload_no_html, timeout=timeout) as response2:
                                response2.raise_for_status()
                                print(f"✅ تم إرسال الرسالة بدون HTML بنجاح")
                                return True
                    
                    response.raise_for_status()
                    print(f"✅ تم إرسال الرسالة بنجاح")
                    return True
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            print(f"⚠️ المحاولة {attempt+1}: فشل إرسال الرسالة - {str(e)}")
            if attempt < max_retries - 1:
                await asyncio.sleep((attempt + 1) * 2)
            else:
                print("❌ فشل إرسال الرسالة بعد جميع المحاولات")
                return False
        except Exception as e:
            print(f"❌ خطأ غير متوقع في إرسال الرسالة: {str(e)}")
            return False
    
    return False

async def send_telegram_photo_async(photo_path, caption=""):
    """إرسال صورة إلى Telegram (نسخة غير متزامنة)"""
    if not TELEGRAM_TOKEN or not AUTHORIZED_USER_ID:
        print("❌ لم يتم تعيين متغيرات Telegram المطلوبة")
        return False
    
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    
    try:
        # قراءة الصورة كبايتس
        with open(photo_path, 'rb') as photo:
            photo_data = photo.read()
        
        # إنشاء نموذج multipart/form-data يدوياً
        form_data = aiohttp.FormData()
        form_data.add_field('chat_id', str(AUTHORIZED_USER_ID))  # تحويل إلى string
        form_data.add_field('photo', photo_data, filename=os.path.basename(photo_path))
        if caption:
            form_data.add_field('caption', str(caption))  # تحويل إلى string
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=form_data) as response:
                response.raise_for_status()
                print(f"✅ تم إرسال الصورة بنجاح: {caption}")
                return True
                
    except Exception as e:
        print(f"❌ فشل إرسال الصورة: {str(e)}")
        return False

def clean_symbol_for_exchange(symbol):
    """تنظيف رمز العملة لجعله متوافقًا مع Binance"""
    # إزالة أي أحرف غير مرغوب فيها
    clean_symbol = symbol.replace("/", "").replace("_", "").upper()
    
    # التأكد من أن الرمز ينتهي بـ USDT
    if not clean_symbol.endswith("USDT"):
        clean_symbol += "USDT"
        
    return clean_symbol

# ====== إعدادات التداول ======
TRADING_MODE = "real"  # "real" أو "demo"
RISK_PER_TRADE = 0.02  # 2% مخاطرة لكل صفقة
import json
import os
# ملف جديد: enhanced_managers.py
import asyncio
from datetime import datetime


class TradeSyncManager:
    def __init__(self, exchange, open_trades_ref):
        self.exchange = exchange
        self.open_orders_cache = {}
        self.open_trades = open_trades_ref  # reference to the main open_trades dict
        
    async def sync_with_exchange(self, symbol):
        """مزامنة الصفقات المحلية مع البورصة"""
        try:
            # جلب جميع الأوامر النشطة من البورصة
            open_orders = await self.exchange.fetch_open_orders(symbol)
            
            # تحديث الذاكرة المؤقتة
            self.open_orders_cache[symbol] = {order['id']: order for order in open_orders}
            
            # البحث عن صفقات مغلقة يدوياً
            closed_manually = []
            for trade_id in list(self.open_trades.keys()):
                if trade_id not in self.open_orders_cache[symbol]:
                    closed_manually.append(trade_id)
            
            # معالجة الصفقات المغلقة يدوياً
            for trade_id in closed_manually:
                await self.handle_manually_closed_trade(trade_id)
                
            return True
            
        except Exception as e:
            print(f"❌ خطأ في المزامنة: {str(e)}")
            return False
    
    async def handle_manually_closed_trade(self, trade_id):
        """معالجة الصفقات المغلقة يدوياً"""
        try:
            if trade_id in self.open_trades:
                trade = self.open_trades[trade_id]
                
                # جلب سعر الإغلاق الفعلي
                ticker = await self.exchange.fetch_ticker(trade['symbol'])
                close_price = ticker['last']
                
                # حساب الربح/الخسارة الفعلي
                # سنحتاج إلى تمرير reference إلى profit calculator
                pnl = {"net": 0, "roe": 0}  # قيم افتراضية مؤقتة
                
                # تحديث السجلات - هذه الدالة يجب أن تكون موجودة في AutoTradeManager
                print(f"📝 تم تحديث سجل الصفقة {trade_id} بالإغلاق اليدوي")
                
                # إرسال إشعار
                await self.notify_manual_closure(trade, close_price, pnl)
                
                # إزالة من القائمة النشطة
                del self.open_trades[trade_id]
                
        except Exception as e:
            print(f"❌ خطأ في معالجة الصفقة المغلقة: {str(e)}")
    
    async def notify_manual_closure(self, trade, close_price, pnl):
        """إشعار الإغلاق اليدوي (مثال أساسي)"""
        message = f"🔓 صفقة مغلقة يدوياً: {trade['symbol']} | السعر: {close_price} | الربح: {pnl['net']:.2f}"
        print(message)
        # هنا يمكنك إضافة إرسال إلى التلغرام أو أي نظام إشعارات آخر

class ProfitCalculator:
    def __init__(self, exchange):
        self.exchange = exchange
        
    async def calculate_actual_pnl(self, trade, exit_price):
        """حساب الربح/الخسارة الفعلي مع مراعاة الرافعة والرسوم"""
        try:
            symbol = trade['symbol']
            side = trade['side']
            entry_price = trade['entry_price']
            size = trade['size']
            leverage = trade.get('leverage', 1)
            
            # الحصول على معلومات التكلفة والرسوم
            market = self.exchange.market(symbol)
            contract_size = market.get('contractSize', 1)
            
            # حساب الربح/الخسارة الأساسي
            if side == 'BUY':
                raw_pnl = (exit_price - entry_price) * size * contract_size
            else:
                raw_pnl = (entry_price - exit_price) * size * contract_size
            
            # تطبيق الرافعة المالية
            leveraged_pnl = raw_pnl * leverage
            
            # حساب الرسوم (دخول وخروج)
            entry_fee = await self.calculate_fee(symbol, size, entry_price)
            exit_fee = await self.calculate_fee(symbol, size, exit_price)
            total_fee = entry_fee + exit_fee
            
            # صافي الربح/الخسارة
            net_pnl = leveraged_pnl - total_fee
            
            return {
                'raw': raw_pnl,
                'leveraged': leveraged_pnl,
                'fees': total_fee,
                'net': net_pnl,
                'roe': (net_pnl / (size * entry_price)) * 100
            }
            
        except Exception as e:
            print(f"❌ خطأ في حساب الأرباح: {str(e)}")
            return None
    
    async def calculate_fee(self, symbol, size, price):
        """حساب الرسوم الدقيقة"""
        try:
            # Binance تفرض رسوم 0.04% على العقود الآجلة
            fee_rate = 0.0004
            trade_value = size * price
            return trade_value * fee_rate
            
        except Exception as e:
            print(f"❌ خطأ في حساب الرسوم: {str(e)}")
            return 0
        

class EnhancedOrderManager:
    def __init__(self, exchange):
        self.exchange = exchange
        
    async def place_guaranteed_sl_tp_order(self, symbol, side, size, entry_price, 
                                         sl_price, tp_price, leverage=1):
        """وضع أوامر مضمونة لوقف الخسارة وجني الأرباح"""
        try:
            # تنظيف رمز الزوج
            clean_symbol = symbol.replace("/", "")
            
            # تحديد نوع الأمر الجانبي
            sl_side = "SELL" if side == "BUY" else "BUY"
            
            # 1. وضع أمر وقف الخسارة (Stop Market)
            sl_order = await self.exchange.create_order(
                clean_symbol, 
                'STOP_MARKET', 
                sl_side, 
                size, 
                None,
                {
                    'stopPrice': sl_price,
                    'reduceOnly': True,
                    'leverage': leverage
                }
            )
            
            # 2. وضع أوامر جني الأرباح (Take Profit Market)
            tp_order = await self.exchange.create_order(
                clean_symbol,
                'TAKE_PROFIT_MARKET',
                sl_side,
                size,
                None,
                {
                    'stopPrice': tp_price,
                    'reduceOnly': True,
                    'leverage': leverage
                }
            )
            
            # 3. متابعة حالة الأوامر
            asyncio.create_task(self.monitor_order_execution(sl_order['id'], tp_order['id'], symbol))
            
            return {
                'sl_order': sl_order,
                'tp_order': tp_order
            }
            
        except Exception as e:
            print(f"❌ خطأ في وضع الأوامر: {str(e)}")
            # محاولة بديلة
            return await self.alternative_order_placement(symbol, side, size, sl_price, tp_price, leverage)
    
    async def monitor_order_execution(self, sl_order_id, tp_order_id, symbol):
        """مراقبة تنفيذ الأوامر وإعادة وضعها إذا لزم الأمر"""
        while True:
            try:
                # التحقق من حالة الأوامر
                sl_status = await self.exchange.fetch_order(sl_order_id, symbol)
                tp_status = await self.exchange.fetch_order(tp_order_id, symbol)
                
                # إذا تم إلغاء أي أمر، إعادة وضعه
                if sl_status['status'] == 'canceled':
                    print("🔄 إعادة وضع أمر وقف الخسارة الملغى")
                    # إعادة وضع الأمر هنا
                    
                if tp_status['status'] == 'canceled':
                    print("🔄 إعادة وضع أمر جني الأرباح الملغى")
                    # إعادة وضع الأمر هنا
                
                await asyncio.sleep(60)  # التحقق كل دقيقة
                
            except Exception as e:
                print(f"❌ خطأ في مراقبة الأوامر: {str(e)}")
                await asyncio.sleep(60)


class OrderManager:
    def __init__(self, exchange):
        self.exchange = exchange

    async def place_guaranteed_sl_tp_order(
        self, symbol, side, size, entry_price, sl_price, tp_price, leverage=1
    ):
        """
        وضع أوامر وقف خسارة (SL) وجني أرباح (TP) مضمونة التنفيذ.
        
        Args:
            symbol (str): زوج التداول.
            side (str): 'BUY' أو 'SELL'.
            size (float): حجم الصفقة.
            entry_price (float): سعر الدخول.
            sl_price (float): سعر وقف الخسارة.
            tp_price (float): سعر جني الأرباح.
            leverage (int, optional): الرافعة المالية. Default is 1.
        
        Returns:
            dict: يحتوي على تفاصيل أوامر SL و TP إذا تم التنفيذ بنجاح.
        """
        try:
            # وضع أمر وقف الخسارة (OCO)
            oco_order = await self.exchange.create_order(
                symbol, 'STOP_LOSS_LIMIT', side, size, sl_price, {
                    'stopPrice': sl_price,
                    'limitPrice': sl_price * 0.995,  # أقل قليلاً لضمان التنفيذ
                    'leverage': leverage,
                    'type': 'STOP_LOSS_LIMIT'
                }
            )

            # وضع أمر جني الأرباح
            tp_order = await self.exchange.create_order(
                symbol, 'TAKE_PROFIT_LIMIT', side, size, tp_price, {
                    'stopPrice': tp_price,
                    'limitPrice': tp_price * 0.995,  # أقل قليلاً لضمان التنفيذ
                    'leverage': leverage,
                    'type': 'TAKE_PROFIT_LIMIT'
                }
            )

            # متابعة حالة الأوامر بشكل غير متزامن
            asyncio.create_task(self.monitor_orders_status(oco_order['id'], tp_order['id']))

            return {
                'sl_order': oco_order,
                'tp_order': tp_order
            }

        except Exception as e:
            print(f"خطأ في وضع الأوامر: {str(e)}")
            return None

    async def monitor_orders_status(self, sl_order_id, tp_order_id):
        """
        مراقبة حالة أوامر SL و TP وتجديدها أو إلغاء الآخر عند التنفيذ.
        
        Args:
            sl_order_id: معرف أمر وقف الخسارة.
            tp_order_id: معرف أمر جني الأرباح.
        """
        while True:
            try:
                sl_status = await self.exchange.fetch_order(sl_order_id)
                tp_status = await self.exchange.fetch_order(tp_order_id)

                # إذا تم إغلاق أحد الأوامر، إلغاء الآخر
                if sl_status['status'] == 'closed' or tp_status['status'] == 'closed':
                    if sl_status['status'] != 'closed':
                        await self.exchange.cancel_order(sl_order_id)
                    if tp_status['status'] != 'closed':
                        await self.exchange.cancel_order(tp_order_id)
                    break

                await asyncio.sleep(30)  # التحقق كل 30 ثانية

            except Exception as e:
                print(f"خطأ في مراقبة الأوامر: {str(e)}")
                await asyncio.sleep(30)

class RiskManagementSystem:
    def __init__(self, exchange, initial_balance, risk_per_trade=0.02):
        self.exchange = exchange
        self.balance = initial_balance
        self.risk_per_trade = risk_per_trade
        self.open_positions = {}
        
    async def calculate_position_size(self, symbol, entry_price, stop_loss_price):
        """حجم المركز الآمن بناء على إدارة المخاطر"""
        try:
            # حساب المخاطرة بالدولار
            risk_amount = self.balance * self.risk_per_trade
            
            # حساب المسافة بين السعر ووقف الخسارة
            if entry_price > stop_loss_price:
                risk_distance = entry_price - stop_loss_price
            else:
                risk_distance = stop_loss_price - entry_price
                
            # حساب حجم المركز
            position_size = risk_amount / risk_distance
            
            # تطبيق حدود الرافعة المالية
            max_leverage = await self.get_max_leverage(symbol)
            max_position_size = (self.balance * max_leverage) / entry_price
            
            return min(position_size, max_position_size)
            
        except Exception as e:
            print(f"خطأ في حساب حجم المركز: {str(e)}")
            return None
    
    async def execute_safe_trade(self, symbol, side, entry_price, sl_price, tp_price, leverage=1):
        """تنفيذ صفقة آمنة مع إدارة مخاطر كاملة"""
        try:
            # 1. حساب حجم المركز الآمن
            size = await self.calculate_position_size(symbol, entry_price, sl_price)
            if size is None:
                return None
            
            # 2. وضع أمر الدخول
            order = await self.exchange.create_order(symbol, 'MARKET', side, size)
            
            # 3. وضع أوامر الوقف والجني
            sl_tp_orders = await self.place_guaranteed_sl_tp_order(
                symbol, side, size, entry_price, sl_price, tp_price, leverage
            )
            
            # 4. تسجيل الصفقة في النظام
            self.open_positions[order['id']] = {
                'symbol': symbol,
                'side': side,
                'size': size,
                'entry_price': entry_price,
                'sl_price': sl_price,
                'tp_price': tp_price,
                'leverage': leverage,
                'sl_order_id': sl_tp_orders['sl_order']['id'],
                'tp_order_id': sl_tp_orders['tp_order']['id'],
                'timestamp': time.time()
            }
            
            return order
            
        except Exception as e:
            print(f"خطأ في تنفيذ الصفقة الآمنة: {str(e)}")
            return None


def calculate_smart_stop_loss(side, current_price, key_levels, volatility=0.01):
    """
    حساب وقف الخسارة الذكي بناءً على مستويات الدعم والمقاومة
    """
    if side == "BUY":
        # لصفقات الشراء: استخدام أقوى مستوى دعم تحت السعر الحالي
        if key_levels.get('strong_support'):
            valid_supports = [s for s in key_levels['strong_support'] if s < current_price]
            if valid_supports:
                strongest_support = max(valid_supports)  # أقوى دعم (أعلى سعر)
                stop_loss = strongest_support * (1 - volatility/2)  # تحت الدعم بقليل
                return stop_loss
        
        # إذا لم يوجد دعم مناسب، استخدام نسبة احتياطية (ولكن ليست ثابتة)
        # حساب نسبة مئوية ديناميكية بناءً على التقلبات
        dynamic_sl_percentage = max(0.01, min(0.03, volatility * 2))
        return current_price * (1 - dynamic_sl_percentage)
    
    else:  # SELL
        # لصفقات البيع: استخدام أقوى مستوى مقاومة فوق السعر الحالي
        if key_levels.get('strong_resistance'):
            valid_resistances = [r for r in key_levels['strong_resistance'] if r > current_price]
            if valid_resistances:
                strongest_resistance = min(valid_resistances)  # أقوى مقاومة (أقل سعر)
                stop_loss = strongest_resistance * (1 + volatility/2)  # فوق المقاومة بقليل
                return stop_loss
        
        # إذا لم توجد مقاومة مناسبة، استخدام نسبة احتياطية
        dynamic_sl_percentage = max(0.01, min(0.03, volatility * 2))
        return current_price * (1 + dynamic_sl_percentage)


class AutoTradeManager:
    def __init__(self, exchange, min_confidence=0.7, risk_per_trade=0.02, leverage=10):
        self.exchange = exchange
        self.min_confidence = min_confidence
        self.risk_per_trade = risk_per_trade
        self.leverage = leverage
        self.auto_trading_enabled = False
        self.open_trades = {}
        self.cache = {}
        self.state_file = "trading_state.json"
        self.trading_fee_rate = 0.0004  # رسوم التداول في Binance (0.04%)
        
        self._auto_update_task = None
        
        # تحميل الحالة السابقة
        self.load_state()

        # إضافة المديرين الجدد
        self.trade_sync = TradeSyncManager(exchange, self.open_trades)
        self.profit_calculator = ProfitCalculator(exchange)
        self.trade_sync.auto_trade_manager = self  

        self._monitoring_tasks = []
        self._is_shutting_down = False
        self.startup_time = datetime.now()

        
        # بدء المزامنة بعد تأخير قصير عندما يكون event loop نشطاً
        self._sync_started = False
        



    async def start_monitoring_system(self):
        """بدء نظام مراقبة الصفقات التلقائي"""
        try:
            # التأكد من بدء المزامنة أولاً
            await self.ensure_sync_started()
            
            # بدء نظام المراقبة في الخلفية
            monitor_task = asyncio.create_task(self.monitor_and_report_trades())
            self._monitoring_tasks.append(monitor_task)
            
            # بدء التحديث التلقائي للصفقات
            update_task = asyncio.create_task(self.auto_update_trades_status())
            self._monitoring_tasks.append(update_task)
            
            # بدء التقرير الساعي
            report_task = asyncio.create_task(self.send_hourly_detailed_report())
            self._monitoring_tasks.append(report_task)
            
            print("✅ بدء نظام مراقبة الصفقات التلقائي")
            
        except Exception as e:
            print(f"❌ خطأ في بدء نظام المراقبة: {str(e)}")
   
    async def stop_all_tasks(self):
        """إيقاف جميع المهام التلقائية بشكل صحيح"""
        self._is_shutting_down = True
        
        # إلغاء جميع المهام
        for task in self._monitoring_tasks:
            if not task.done():
                task.cancel()
        
        # انتظار انتهاء جميع المهام
        if self._monitoring_tasks:
            await asyncio.gather(*self._monitoring_tasks, return_exceptions=True)
        
        self._monitoring_tasks = []
        print("✅ تم إيقاف جميع المهام التلقائية")

    

    async def monitor_and_report_trades(self, report_interval_minutes=30):
        """
        مراقبة الصفقات النشطة وإرسال تقارير دورية إلى Telegram
        """
        try:
            print("📊 بدء نظام مراقبة الصفقات والتقارير الدورية")
            
            while not self._is_shutting_down:
                try:
                    # التحقق من وجود صفقات نشطة
                    if not self.open_trades:
                        print("⚠️ لا توجد صفقات نشطة للمراقبة، الانتظار...")
                        # الانتظار مع التحقق من الإغلاق
                        for _ in range(report_interval_minutes * 60):
                            if self._is_shutting_down:
                                break
                            await asyncio.sleep(1)
                        continue

                    # مزامنة الصفقات مع Binance أولاً
                    print("🔄 مزامنة الصفقات مع Binance...")
                    await self.sync_with_binance()
                    
                    # جمع معلومات الصفقات النشطة
                    active_trades = []
                    profit_trades = []
                    loss_trades = []
                    
                    for trade_id, trade in list(self.open_trades.items()):
                        try:
                            # الحصول على السعر الحالي
                            current_price = await self.get_current_price(trade['symbol'])
                            if current_price <= 0:
                                continue
                            
                            # حساب الربح/الخسارة الحالية
                            pnl_data = await self.calculate_trade_pnl(trade, current_price)
                            
                            # تتبع حالة الصفقة
                            trade_info = {
                                'symbol': trade['symbol'],
                                'side': trade['side'],
                                'entry_price': trade['entry_price'],
                                'current_price': current_price,
                                'pnl': pnl_data['net'],
                                'pnl_percentage': pnl_data['roe'],
                                'sl_level': trade['sl_level'],
                                'tp_levels': trade['tp_levels'],
                                'size': trade['size'],
                                'trade_id': trade_id
                            }
                            
                            active_trades.append(trade_info)
                            
                            # التحقق من تحقيق أهداف الربح
                            tp_hit = False
                            for i, tp_level in enumerate(trade['tp_levels']):
                                if ((trade['side'] == 'BUY' and current_price >= tp_level) or
                                    (trade['side'] == 'SELL' and current_price <= tp_level)):
                                    if not trade.get(f'tp_{i}_hit', False):
                                        # إرسال إشعار تحقيق هدف
                                        await self.send_tp_notification(trade, tp_level, i+1, current_price)
                                        trade[f'tp_{i}_hit'] = True
                                        profit_trades.append(trade_info)
                                        tp_hit = True
                            
                            # التحقق من وقف الخسارة
                            if ((trade['side'] == 'BUY' and current_price <= trade['sl_level']) or
                                (trade['side'] == 'SELL' and current_price >= trade['sl_level'])):
                                if not trade.get('sl_hit', False):
                                    # إرسال إشعار وقف الخسارة
                                    await self.send_sl_notification(trade, current_price)
                                    trade['sl_hit'] = True
                                    loss_trades.append(trade_info)
                                    
                        except Exception as e:
                            print(f"❌ خطأ في مراقبة الصفقة {trade_id}: {str(e)}")
                            continue
                    
                    # إرسال التقرير الدوري
                    if active_trades:
                        await self.send_periodic_report(active_trades, profit_trades, loss_trades)
                    
                    # الانتظار للفترة المحددة مع التحقق من الإغلاق
                    print(f"⏰ الانتظار {report_interval_minutes} دقائق للتقرير التالي...")
                    for _ in range(report_interval_minutes * 60):
                        if self._is_shutting_down:
                            break
                        await asyncio.sleep(1)
                        
                except Exception as e:
                    print(f"❌ خطأ في دورة المراقبة: {str(e)}")
                    # الانتظار دقيقة قبل إعادة المحاولة
                    for _ in range(60):
                        if self._is_shutting_down:
                            break
                        await asyncio.sleep(1)
                        
        except Exception as e:
            print(f"❌ خطأ غير متوقع في مراقبة الصفقات: {str(e)}")

    async def auto_update_trades_status(self):
        """
        تحديث حالة الصفقات تلقائياً مع التحقق من حالة الإغلاق
        """
        try:
            print("🔄 بدء التحديث التلقائي لحالة الصفقات")
            
            while not self._is_shutting_down:
                try:
                    if not self.open_trades:
                        # الانتظار مع التحقق من الإغلاق
                        for _ in range(300):
                            if self._is_shutting_down:
                                break
                            await asyncio.sleep(1)
                        continue
                    
                    # نسخ القائمة لتجنب تغييرها أثناء التكرار
                    trades_to_check = list(self.open_trades.items())
                    
                    for trade_id, trade in trades_to_check:
                        if self._is_shutting_down:
                            break
                            
                        try:
                            current_price = await self.get_current_price(trade['symbol'])
                            
                            # التحقق إذا تم إغلاق الصفقة (وصل إلى STOP LOSS)
                            if ((trade['side'] == "BUY" and current_price <= trade['sl_level']) or
                                (trade['side'] == "SELL" and current_price >= trade['sl_level'])):
                                
                                # حساب PNL النهائي
                                if trade['side'] == "BUY":
                                    pnl = (current_price - trade['entry_price']) * trade['size']
                                else:
                                    pnl = (trade['entry_price'] - current_price) * trade['size']
                                
                                # إرسال إشعار الإغلاق
                                await send_telegram_message_async(
                                    f"⚠️ **تم إغلاق الصفقة تلقائياً**\n"
                                    f"• الزوج: {trade['symbol']}\n"
                                    f"• المعرف: {trade_id}\n"
                                    f"• السبب: وصل إلى وقف الخسارة\n"
                                    f"• الأرباح/الخسائر النهائية: {pnl:+.2f} USDT\n"
                                    f"• السعر النهائي: {current_price:.6f}"
                                )
                                
                                # إزالة الصفقة من القائمة
                                if trade_id in self.open_trades:
                                    del self.open_trades[trade_id]
                                    self.save_state()
                            
                            # التحقق من تحقيق جميع الأهداف
                            else:
                                achieved_targets = 0
                                for tp_level in trade['tp_levels']:
                                    if ((trade['side'] == "BUY" and current_price >= tp_level) or
                                        (trade['side'] == "SELL" and current_price <= tp_level)):
                                        achieved_targets += 1
                                
                                # إذا تم تحقيق جميع الأهداف
                                if achieved_targets == len(trade['tp_levels']):
                                    if trade['side'] == "BUY":
                                        pnl = (current_price - trade['entry_price']) * trade['size']
                                    else:
                                        pnl = (trade['entry_price'] - current_price) * trade['size']
                                    
                                    await send_telegram_message_async(
                                        f"🎯 **تم تحقيق جميع الأهداف**\n"
                                        f"• الزوج: {trade['symbol']}\n"
                                        f"• المعرف: {trade_id}\n"
                                        f"• الأرباح النهائية: {pnl:+.2f} USDT\n"
                                        f"• السعر النهائي: {current_price:.6f}"
                                    )
                                    
                                    if trade_id in self.open_trades:
                                        del self.open_trades[trade_id]
                                        self.save_state()
                        
                        except Exception as e:
                            print(f"❌ خطأ في التحقق من الصفقة {trade_id}: {str(e)}")
                            continue
                    
                    # الانتظار 5 دقائق قبل التحديث التالي مع التحقق من الإغلاق
                    print("⏰ الانتظار 5 دقائق للتحديث التالي...")
                    for _ in range(300):
                        if self._is_shutting_down:
                            break
                        await asyncio.sleep(1)
                        
                except Exception as e:
                    print(f"❌ خطأ في التحديث التلقائي: {str(e)}")
                    # الانتظار دقيقة قبل إعادة المحاولة مع التحقق من الإغلاق
                    for _ in range(60):
                        if self._is_shutting_down:
                            break
                        await asyncio.sleep(1)
                        
        except Exception as e:
            print(f"❌ خطأ غير متوقع في التحديث التلقائي: {str(e)}")


    async def send_hourly_detailed_report(self):
        """
        إرسال تقرير مفصل كل ساعة
        """
        try:
            print("📈 بدء إرسال التقارير الساعية المفصلة")
            
            while not self._is_shutting_down:
                try:
                    # الحصول على معلومات مفصلة من Binance
                    detailed_info = await self.get_detailed_trade_info()
                    
                    if not detailed_info:
                        print("⚠️ لا توجد معلومات مفصلة للإبلاغ")
                        # الانتظار ساعة مع التحقق من الإغلاق
                        for _ in range(3600):
                            if self._is_shutting_down:
                                break
                            await asyncio.sleep(1)
                        continue
                    
                    # بناء رسالة التقرير المفصل
                    message = "📋 **تقرير مفصل عن صفقات Binance**\n\n"
                    
                    for info in detailed_info:
                        message += (
                            f"🎯 **{info['symbol']}**\n"
                            f"• الأوامر المفتوحة: {info['open_orders']}\n"
                            f"• التداولات الأخيرة (24h): {info['recent_trades']}\n"
                        )
                        
                        if info['orders']:
                            message += "<b>أحدث الأوامر:</b>\n"
                            for order in info['orders'][:3]:  # عرض آخر 3 أوامر فقط
                                message += (
                                    f"• {order['side']} | السعر: {order['price']} | "
                                    f"الكمية: {order['amount']} | المنفذ: {order['filled']}\n"
                                )
                        
                        message += "\n"
                    
                    # إضافة معلومات الصفقات المحلية
                    if self.open_trades:
                        message += "📊 **الصفقات المحلية النشطة:**\n"
                        for trade_id, trade in self.open_trades.items():
                            message += (
                                f"• {trade['symbol']} | {trade['side']} | "
                                f"الدخول: {trade['entry_price']} | "
                                f"الحجم: {trade['size']}\n"
                            )
                    
                    message += f"\n⏰ **وقت التقرير:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    
                    # إرسال التقرير
                    await send_telegram_message_async(message)
                    print("✅ تم إرسال التقرير الساعي المفصل")
                    
                    # الانتظار ساعة مع التحقق من الإغلاق
                    print("⏰ الانتظار ساعة للتقرير التالي...")
                    for _ in range(3600):
                        if self._is_shutting_down:
                            break
                        await asyncio.sleep(1)
                        
                except Exception as e:
                    print(f"❌ خطأ في التقرير الساعي: {str(e)}")
                    # الانتظار 5 دقائق قبل إعادة المحاولة
                    for _ in range(300):
                        if self._is_shutting_down:
                            break
                        await asyncio.sleep(1)
                        
        except Exception as e:
            print(f"❌ خطأ غير متوقع في التقرير الساعي: {str(e)}")



    async def ensure_sync_started(self):
        """التأكد من بدء المزامنة (استدعاء هذه الدالة عند الحاجة)"""
        if not self._sync_started:
            asyncio.create_task(self.periodic_sync())
            self._sync_started = True
    
    async def periodic_sync(self):
        """مزامنة دورية مع البورصة"""
        while True:
            try:
                # الحصول على جميع الرموز الفريدة من الصفقات المفتوحة
                symbols = set(trade['symbol'] for trade in self.open_trades.values())
                
                # استخدام asyncio.gather لتشغيل المهام بشكل متوازي
                tasks = []
                for symbol in symbols:
                    # تأكد من أن sync_with_exchange هي دالة غير متزامنة
                    task = asyncio.create_task(self.trade_sync.sync_with_exchange(symbol))
                    tasks.append(task)
                
                # انتظار انتهاء جميع المهام
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                
                await asyncio.sleep(300)  # مزامنة كل 5 دقائق
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"❌ خطأ في المزامنة الدورية: {str(e)}")
                await asyncio.sleep(300)

    
    async def handle_manually_closed_trade(self, trade_id):
        """معالجة الصفقات المغلقة يدوياً مع حساب الأرباح الجزئية"""
        try:
            if trade_id in self.open_trades:
                trade = self.open_trades[trade_id]
                
                # جلب سعر الإغلاق الفعلي
                ticker = await self.exchange.fetch_ticker(trade['symbol'])
                close_price = ticker['last']
                
                # استخدام نظام وقف الخسارة الجديد لحساب الأرباح
                await self.handle_stop_loss(trade_id, close_price)
                    
        except Exception as e:
            print(f"❌ خطأ في معالجة الصفقة المغلقة: {str(e)}")


    async def _handle_stop_loss_fallback(self, trade_id, close_price):
        """طريقة بديلة إذا لم يكن AutoTradeManager متاحاً"""
        try:
            if trade_id in self.open_trades:
                trade = self.open_trades[trade_id]
                
                # حساب الربح/الخسارة الأساسي
                if trade['side'] == 'BUY':
                    pnl_pct = ((close_price - trade['entry_price']) / trade['entry_price']) * 100
                else:
                    pnl_pct = ((trade['entry_price'] - close_price) / trade['entry_price']) * 100
                
                # حساب صافي الربح/الخسارة مع مراعاة الأرباح الجزئية
                total_profit = 0
                if 'partial_profits' in trade:
                    for partial in trade['partial_profits']:
                        total_profit += partial['profit']
                
                # حساب ربح/خسارة الجزء المتبقي
                remaining_size = trade['size']
                if 'partial_profits' in trade:
                    for partial in trade['partial_profits']:
                        remaining_size -= partial.get('closed_size', 0)
                
                if remaining_size > 0:
                    gross_profit = remaining_size * pnl_pct / 100
                    fee = gross_profit * 0.0004  # رسوم Binance
                    net_profit = gross_profit - fee
                    total_profit += net_profit
                
                # إرسال إشعار
                message = f"🔓 صفقة مغلقة يدوياً: {trade['symbol']} | السعر: {close_price} | الربح: {total_profit:.2f}"
                print(message)
                
                # إزالة من القائمة النشطة
                del self.open_trades[trade_id]
                
        except Exception as e:
            print(f"❌ خطأ في المعالجة البديلة: {str(e)}")

    async def stop_periodic_sync(self):
        """إيقاف المزامنة الدورية"""
        if hasattr(self, '_sync_task') and self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass
            self._sync_task = None

    # أضف الوظائف المفقودة
    def update_trade_history(self, trade_id, close_price, pnl, reason):
        """تحديث سجل الصفقات (يجب تطبيقه حسب نظامك)"""
        # TODO: تطبيق هذا الدالة حسب نظام التخزين الخاص بك
        pass
        
    async def notify_manual_closure(self, trade, close_price, pnl):
        """إشعار الإغلاق اليدوي (يجب تطبيقه حسب نظامك)"""
        # TODO: تطبيق هذا الدالة حسب نظام الإشعارات الخاص بك
        pass


    async def sync_with_binance(self, symbol=None):
        """
        مزامنة فورية مع Binance للحصول على أحدث حالة للصفقات المفتوحة
        مع التحقق من الصفقات المغلقة ومعالجتها.
        """
        try:
            # تحديد الرموز المراد مزامنتها
            if symbol:
                symbols_to_sync = [symbol]
            else:
                symbols_to_sync = list(set(trade['symbol'] for trade in self.open_trades.values()))

            for symbol in symbols_to_sync:
                clean_symbol = symbol.replace("/", "")

                # الحصول على جميع الأوامر المفتوحة من Binance
                open_orders = await self.exchange.fetch_open_orders(clean_symbol)
                open_order_ids = {order['id'] for order in open_orders}

                # الحصول على تاريخ الصفقات الأخيرة (آخر 24 ساعة)
                since = int((datetime.now() - timedelta(hours=24)).timestamp() * 1000)
                recent_trades = await self.exchange.fetch_my_trades(clean_symbol, since=since)

                # التحقق من الصفقات المحلية المفتوحة مقابل بيانات Binance
                for trade_id, trade_data in list(self.open_trades.items()):
                    if trade_data['symbol'] != symbol:
                        continue

                    # افتراض أن الصفقة مغلقة حتى يثبت العكس
                    is_closed_in_binance = True

                    # إذا كانت موجودة ضمن الأوامر المفتوحة
                    if trade_id in open_order_ids:
                        is_closed_in_binance = False

                    # إذا تم تنفيذها مؤخرًا ضمن الصفقات الأخيرة
                    for t in recent_trades:
                        if t.get('order') == trade_id and t.get('symbol') == clean_symbol:
                            is_closed_in_binance = False
                            break

                    # معالجة الصفقة المغلقة
                    if is_closed_in_binance:
                        print(f"🔄 اكتشاف صفقة مغلقة في Binance ولكنها مفتوحة محلياً: {trade_id}")
                        await self.handle_binance_closed_trade(trade_id)

                print(f"✅ تمت مزامنة صفقات {symbol} مع Binance بنجاح.")

        except Exception as e:
            print(f"❌ خطأ في المزامنة مع Binance: {e}")


    async def handle_binance_closed_trade(self, trade_id):
        """معالجة الصفقات التي تم إغلاقها في Binance"""
        try:
            if trade_id not in self.open_trades:
                return
                
            trade = self.open_trades[trade_id]
            symbol = trade['symbol']
            clean_symbol = symbol.replace("/", "")
            
            # محاولة الحصول على سعر الإغلاق من تاريخ التداول
            since = int((trade['timestamp'] - timedelta(hours=24)).timestamp() * 1000)
            trades_history = await self.exchange.fetch_my_trades(clean_symbol, since=since)
            
            close_price = None
            for t in trades_history:
                if t['order'] == trade_id:
                    close_price = t['price']
                    break
            
            # إذا لم يتم العثور على سعر الإغلاق، استخدام السعر الحالي
            if close_price is None:
                ticker = await self.exchange.fetch_ticker(clean_symbol)
                close_price = ticker['last']
            
            # حساب الربح/الخسارة
            pnl = await self.calculate_trade_pnl(trade, close_price)
            
            # إرسال إشعار بالإغلاق
            message = (
                f"🔄 <b>تم اكتشاف إغلاق صفقة من Binance</b>\n"
                f"• الزوج: {trade['symbol']}\n"
                f"• المعرف: {trade_id}\n"
                f"• السعر: {close_price:.6f}\n"
                f"• الربح/الخسارة: {pnl['net']:+.2f} USDT\n"
                f"• تمت إزالتها من القائمة المحلية"
            )
            await send_telegram_message_async(message)
            
            # إزالة الصفقة من القائمة المحلية
            del self.open_trades[trade_id]
            self.save_state()
            
        except Exception as e:
            print(f"❌ خطأ في معالجة الصفقة المغلقة من Binance: {str(e)}")

    async def check_trade_active_in_binance(self, trade_id, symbol):
        """التحقق من أن الصفقة لا تزال نشطة في Binance"""
        try:
            clean_symbol = symbol.replace("/", "")
            
            # التحقق من الأوامر المفتوحة
            open_orders = await self.exchange.fetch_open_orders(clean_symbol)
            for order in open_orders:
                if order['id'] == trade_id:
                    return True
            
            # التحقق من تاريخ التداول الأخير
            since = int((datetime.now() - timedelta(hours=24)).timestamp() * 1000)
            trades = await self.exchange.fetch_my_trades(clean_symbol, since=since)
            for trade in trades:
                if trade['order'] == trade_id:
                    return True
            
            return False
            
        except Exception as e:
            print(f"❌ خطأ في التحقق من نشاط الصفقة في Binance: {str(e)}")
            return False

    async def calculate_trade_pnl(self, trade, exit_price):
        """حساب الربح/الخسارة للصفقة"""
        try:
            if trade['side'] == 'BUY':
                pnl = (exit_price - trade['entry_price']) * trade['size']
            else:
                pnl = (trade['entry_price'] - exit_price) * trade['size']
            
            return {
                'net': pnl,
                'roe': (pnl / (trade['entry_price'] * trade['size'])) * 100
            }
        except Exception as e:
            print(f"❌ خطأ في حساب الربح/الخسارة: {str(e)}")
            return {'net': 0, 'roe': 0}


    async def get_available_balance(self):
        """الحصول على الرصيد المتاح للاستخدام"""
        try:
            if TRADING_MODE == "real":
                # استخدام await مع fetch_balance
                balance = await self.exchange.fetch_balance()
                if 'USDT' in balance and 'free' in balance['USDT']:
                    return float(balance['USDT']['free'])
                else:
                    # البحث عن رصيد USDT في البنية المختلفة
                    for key, value in balance.items():
                        if isinstance(value, dict) and 'free' in value and key == 'USDT':
                            return float(value['free'])
                    print("لم يتم العثور على رصيد USDT")
                    return 1000
            else:
                return 1000
        except Exception as e:
            print(f"خطأ في جلب الرصيد: {str(e)}")
            return 1000

    async def get_min_order_size(self, symbol):
        """الحصول على الحد الأدنى لحجم الأمر"""
        try:
            # استخدام await مع load_markets
            markets = await self.exchange.load_markets()
            market = markets.get(symbol.replace("/", ""))
            
            if market and 'limits' in market and 'amount' in market['limits']:
                min_amount = market['limits']['amount'].get('min', 0.001)
                return max(0.001, min_amount)
            
            return 0.001
        except Exception as e:
            print(f"خطأ في جلب الحد الأدنى لحجم الأمر: {str(e)}")
            return 0.001

    async def get_max_order_size(self, symbol, balance):
        """الحصول على الحد الأقصى لحجم الأمر بناء على الرصيد"""
        try:
            # استخدام await مع get_current_price
            price = await self.get_current_price(symbol)
            if price <= 0:
                return 0
                
            max_size = (balance * 0.95) / price
            
            # استخدام await مع load_markets
            markets = await self.exchange.load_markets()
            market = markets.get(symbol.replace("/", ""))
            
            if market and 'limits' in market and 'amount' in market['limits']:
                max_amount = market['limits']['amount'].get('max', float('inf'))
                return min(max_size, max_amount)
                
            return max_size
        except Exception as e:
            print(f"خطأ في جلب الحد الأقصى لحجم الأمر: {str(e)}")
            price = await self.get_current_price(symbol)
            return (balance * 0.95) / price if price > 0 else 0


    def get_cached_data(self, symbol, time_frame, since):
        """التخزين المؤقت للبيانات داخل الفئة"""
        cache_key = f"{symbol}_{time_frame}_{since}"
        
        # محاولة تحميل البيانات من الذاكرة المؤقتة
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # جلب البيانات من API إذا لم توجد في الذاكرة المؤقتة
        try:
            candles = self.exchange.fetch_ohlcv(symbol, timeframe=time_frame, since=since, limit=500)
            if not candles:
                print(f"⚠️ لم يتم العثور على بيانات للزوج {symbol} والإطار {time_frame}")
                return pd.DataFrame()
            
            # تحويل البيانات
            df = pd.DataFrame(candles, columns=["time", "open", "high", "low", "close", "volume"])
            df["time"] = pd.to_datetime(df["time"], unit="ms")
            
            # تخزين البيانات في الذاكرة المؤقتة
            self.cache[cache_key] = df
            return df
            
        except Exception as e:
            print(f"❌ خطأ في جلب البيانات للزوج {symbol}: {str(e)}")
            return pd.DataFrame()



    def load_state(self):
        """
        تحميل حالة التداول السابقة من ملف مع معالجة متقدمة لأخطاء JSON
        """
        try:
            if not os.path.exists(self.state_file):
                print("⚠️ لا يوجد ملف حالة سابق، سيبدأ بحالة فارغة")
                return

            # قراءة محتوى الملف أولاً
            with open(self.state_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()

            # إذا كان الملف فارغاً
            if not content:
                print("⚠️ ملف الحالة فارغ، سيبدأ بحالة فارغة")
                return

            # محاولة إصلاح أخطاء JSON الشائعة
            content = self.fix_json_errors(content)

            try:
                # محاولة تحميل JSON بعد الإصلاح
                state_data = json.loads(content)
            except json.JSONDecodeError as e:
                print(f"❌ فشل تحميل ملف الحالة بعد الإصلاح: {str(e)}")
                
                # إنشاء نسخة احتياطية من الملف التالف
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                backup_file = f"{self.state_file}.corrupted.{timestamp}"
                
                # نسخ المحتوى إلى الملف الاحتياطي
                with open(backup_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                print(f"📦 تم إنشاء نسخة احتياطية من الملف التالف: {backup_file}")
                
                # بدء حالة فارغة بدلاً من محاولة الاستعادة
                self.open_trades = {}
                self.auto_trading_enabled = False
                print("🔄 البدء بحالة فارغة بسبب فشل تحميل الحالة")
                return

            # تحميل البيانات بنجاح
            self.auto_trading_enabled = state_data.get('auto_trading_enabled', False)
            
            # تحميل الصفقات المفتوحة
            self.open_trades = {}
            for trade_id, trade_data in state_data.get('open_trades', {}).items():
                try:
                    # التحقق من وجود جميع الحقول الضرورية
                    required_fields = ['symbol', 'side', 'entry_price', 'tp_levels', 
                                     'sl_level', 'size', 'timestamp']
                    
                    if all(field in trade_data for field in required_fields):
                        self.open_trades[trade_id] = {
                            'symbol': trade_data['symbol'],
                            'side': trade_data['side'],
                            'entry_price': float(trade_data['entry_price']),
                            'tp_levels': [float(tp) for tp in trade_data['tp_levels']],
                            'sl_level': float(trade_data['sl_level']),
                            'size': float(trade_data['size']),
                            'timestamp': datetime.fromisoformat(trade_data['timestamp']),
                            'signal_confidence': float(trade_data.get('signal_confidence', 0.5))
                        }
                    else:
                        print(f"⚠️ بيانات الصفقة {trade_id} ناقصة، سيتم تخطيها")
                        
                except (ValueError, TypeError) as e:
                    print(f"⚠️ خطأ في تحويل بيانات الصفقة {trade_id}: {str(e)}")
                    continue
            
            print(f"✅ تم تحميل {len(self.open_trades)} صفقة نشطة من الحالة السابقة")
            
        except FileNotFoundError:
            print(f"⚠️ ملف الحالة {self.state_file} غير موجود")
        except Exception as e:
            print(f"❌ خطأ غير متوقع في تحميل الحالة: {str(e)}")
            # بدء حالة فارغة في حالة الخطأ
            self.open_trades = {}
            self.auto_trading_enabled = False

    def fix_json_errors(self, content):
        """
        إصلاح الأخطاء الشائعة في تنسيق JSON مع تحسينات متقدمة
        """
        try:
            # إزالة أي أحرف غير UTF-8
            content = content.encode('utf-8', 'ignore').decode('utf-8')
            
            # استبدال الاقتباس المفرد بمزدوج
            content = content.replace("'", '"')
            
            # إزالة الفواصل الزائدة في نهاية الكائنات والمصفوفات
            content = re.sub(r',\s*}', '}', content)
            content = re.sub(r',\s*]', ']', content)
            
            # إصلاح الخطأ المحدد: Expecting ':' delimiter
            # البحث عن نمط المفتاح بدون النقطتين
            content = re.sub(r'("[^"]+")\s*([^"\s{}\]\[,]+)', r'\1: \2', content)
            
            # إضافة اقتباس حول أسماء الخصائص إذا كانت مفقودة
            content = re.sub(r'(\w+)\s*:', r'"\1":', content)
            
            # إصلاح الفواصل الناقصة بين العناصر
            content = re.sub(r'("[^"]*")\s*([^"\s{}\]\[,])', r'\1, \2', content)
            
            # إصلاح الأقواس غير المغلقة
            open_braces = content.count('{')
            close_braces = content.count('}')
            
            if open_braces > close_braces:
                content += '}' * (open_braces - close_braces)
            elif close_braces > open_braces:
                content = content[:content.rfind('}')]  # إزالة الأقواس الزائدة
            
            # التحقق من صحة تنسيق الملف الإلكتروني
            lines = content.split('\n')
            for i, line in enumerate(lines):
                # البحث عن أخطاء في التنسيق وإصلاحها
                if ':' not in line and '"' in line:
                    # إذا كان السطر يحتوي على اقتباس ولكن لا يحتوي على نقطتين
                    parts = line.split('"')
                    if len(parts) >= 3:
                        lines[i] = f'"{parts[1]}": "{parts[3]}"'
            
            content = '\n'.join(lines)
            
            # التحقق النهائي من صحة JSON
            try:
                json.loads(content)
                print("✅ تم إصلاح JSON بنجاح")
            except json.JSONDecodeError as e:
                print(f"⚠️ لا يزال JSON غير صالح بعد الإصلاح: {str(e)}")
                # محاولة إصلاح إضافية
                content = self.advanced_json_repair(content)
            
            return content
            
        except Exception as e:
            print(f"❌ خطأ في إصلاح JSON: {str(e)}")
            return content

    def advanced_json_repair(self, content):
        """
        إصلاح متقدم لأخطاء JSON المستعصية
        """
        try:
            # تحليل المحتوى سطراً سطراً
            lines = content.split('\n')
            repaired_lines = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # إصلاح الأخطاء الشائعة في كل سطر
                if line.endswith(','):
                    line = line[:-1]
                
                if ':' in line:
                    parts = line.split(':', 1)
                    key = parts[0].strip()
                    value = parts[1].strip()
                    
                    # إضافة اقتباس إذا كان المفتاح بدون اقتباس
                    if not key.startswith('"'):
                        key = f'"{key}"'
                    
                    # إضافة اقتباس إذا كانت القيمة بدون اقتباس ولم تكن رقماً
                    if not value.startswith('"') and not value.replace('.', '').isdigit():
                        value = f'"{value}"'
                    
                    line = f'{key}: {value}'
                
                repaired_lines.append(line)
            
            # تجميع الخطوط معاً
            content = '\n'.join(repaired_lines)
            
            # إضافة الأقواس إذا كانت مفقودة
            if not content.startswith('{'):
                content = '{' + content
            if not content.endswith('}'):
                content = content + '}'
            
            return content
            
        except Exception as e:
            print(f"❌ خطأ في الإصلاح المتقدم: {str(e)}")
            return content


    def save_state(self):
        """حفظ حالة التداول الحالية إلى ملف"""
        try:
            # تحويل البيانات إلى تنسيق قابل للتخزين
            state_data = {
                'open_trades': {},
                'auto_trading_enabled': self.auto_trading_enabled,
                'last_updated': datetime.now().isoformat()
            }

            for trade_id, trade in self.open_trades.items():
                # التأكد من أن جميع القيم قابلة للتسلسل
                state_data['open_trades'][trade_id] = {
                    'symbol': str(trade['symbol']),
                    'side': str(trade['side']),
                    'entry_price': float(trade['entry_price']),
                    'tp_levels': [float(tp) for tp in trade['tp_levels']],
                    'sl_level': float(trade['sl_level']),
                    'size': float(trade['size']),
                    'timestamp': trade['timestamp'].isoformat(),
                    'signal_confidence': float(trade['signal_confidence'])
                }

            # حفظ البيانات في ملف
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(state_data, f, indent=4, ensure_ascii=False)

            print(f"✅ تم حفظ حالة التداول في {self.state_file}")

        except Exception as e:
            print(f"❌ خطأ في حفظ حالة التداول: {str(e)}")

    def restore_from_backup(self):
        """
        استعادة الحالة من آخر نسخة احتياطية
        """
        try:
            import glob
            # البحث عن أحدث نسخة احتياطية
            backup_files = glob.glob(f"{self.state_file}.corrupted.*")
            if not backup_files:
                print("⚠️ لا توجد نسخ احتياطية للاستعادة")
                return False
            
            # العثور على أحدث نسخة احتياطية
            latest_backup = max(backup_files, key=os.path.getctime)
            
            # نسخ الملف الاحتياطي إلى ملف الحالة الرئيسي
            import shutil
            
            # إذا كان الملف الرئيسي موجوداً، احذفه أولاً
            if os.path.exists(self.state_file):
                try:
                    os.remove(self.state_file)
                except PermissionError:
                    # إذا لم نتمكن من حذف الملف، استخدم اسمًا مختلفًا
                    temp_file = f"{self.state_file}.temp"
                    shutil.copy2(latest_backup, temp_file)
                    latest_backup = temp_file
            
            shutil.copy2(latest_backup, self.state_file)
            print(f"✅ تم استعادة الحالة من النسخة الاحتياطية: {latest_backup}")
            
            # إعادة تحميل الحالة
            self.load_state()
            return True
            
        except Exception as e:
            print(f"❌ خطأ في استعادة النسخة الاحتياطية: {str(e)}")
            return False



    async def verify_active_trades(self):
        """التحقق من الصفقات النشطة مع البورصة والتأكد من صحتها"""
        try:
            trades_to_remove = []
            
            for trade_id, trade in self.open_trades.items():
                try:
                    # الحصول على وضع الصفقة من البورصة
                    order_status = await self.get_order_status(trade_id, trade['symbol'])
                    
                    if order_status in ['closed', 'canceled', 'expired']:
                        # الصفقة مغلقة، يجب إزالتها من القائمة
                        trades_to_remove.append(trade_id)
                        print(f"⚠️ الصفقة {trade_id} مغلقة في البورصة، سيتم إزالتها")
                    
                    # تحديث سعر الدخول والحجم إذا لزم الأمر
                    current_price = await self.get_current_price(trade['symbol'])
                    if current_price:
                        # يمكنك هنا تحديث أي معلومات تحتاج للتحديث
                        pass
                        
                except Exception as e:
                    print(f"❌ خطأ في التحقق من الصفقة {trade_id}: {str(e)}")
                    # في حالة الخطأ، نعتبر أن الصفقة لا تزال نشطة
            
            # إزالة الصفقات المغلقة
            for trade_id in trades_to_remove:
                if trade_id in self.open_trades:
                    del self.open_trades[trade_id]
            
            # حفظ الحالة بعد التحديث
            self.save_state()
            
        except Exception as e:
            print(f"❌ خطأ عام في التحقق من الصفقات النشطة: {str(e)}")
    


    async def get_order_status(self, order_id, symbol):
        """الحصول على حالة الصفقة من البورصة"""
        try:
            # تنظيف رمز الزوج
            clean_symbol = symbol.replace("/", "")
            
            # محاولة الحصول على معلومات الصفقة
            order = self.exchange.fetch_order(order_id, clean_symbol)
            return order.get('status', 'unknown')
            
        except Exception as e:
            print(f"❌ خطأ في الحصول على حالة الصفقة {order_id}: {str(e)}")
            return 'unknown'
        
   
  
    def _calculate_support_resistance_sl(self, side, entry_price, df, volatility=0.01):
        """
        حساب وقف الخسارة بناءً على مناطق الدعم والمقاومة القريبة
        """
        try:
            # تحليل آخر 50 شمعة للعثور على مستويات الدعم والمقاومة
            lookback = min(50, len(df))
            recent_data = df.iloc[-lookback:]
            
            # حساب المتوسط المتحرك لاستخدامه كمرجع
            ma_20 = recent_data['close'].rolling(20).mean().iloc[-1]
            
            if side == "BUY":
                # البحث عن أقوى مستوى دعم تحت السعر الحالي
                support_levels = []
                for i in range(2, lookback-2):
                    if (recent_data['low'].iloc[i] < recent_data['low'].iloc[i-1] and
                        recent_data['low'].iloc[i] < recent_data['low'].iloc[i-2] and
                        recent_data['low'].iloc[i] < recent_data['low'].iloc[i+1] and
                        recent_data['low'].iloc[i] < recent_data['low'].iloc[i+2]):
                        support_levels.append(recent_data['low'].iloc[i])
                
                if support_levels:
                    # أخذ أقوى مستوى دعم تحت سعر الدخول
                    valid_supports = [s for s in support_levels if s < entry_price]
                    if valid_supports:
                        strongest_support = max(valid_supports)
                        # جعل وقف الخسارة تحت مستوى الدعم بقليل
                        sl_price = strongest_support * (1 - volatility/2)
                        return max(sl_price, entry_price * (1 - volatility * 3))
                
                # إذا لم نجد دعم قوي، نستخدم المتوسط المتحرك أو نسبة ثابتة
                return min(entry_price * (1 - volatility * 2), ma_20 * (1 - volatility))
            
            else:  # SELL
                # البحث عن أقوى مستوى مقاومة فوق السعر الحالي
                resistance_levels = []
                for i in range(2, lookback-2):
                    if (recent_data['high'].iloc[i] > recent_data['high'].iloc[i-1] and
                        recent_data['high'].iloc[i] > recent_data['high'].iloc[i-2] and
                        recent_data['high'].iloc[i] > recent_data['high'].iloc[i+1] and
                        recent_data['high'].iloc[i] > recent_data['high'].iloc[i+2]):
                        resistance_levels.append(recent_data['high'].iloc[i])
                
                if resistance_levels:
                    # أخذ أقوى مستوى مقاومة فوق سعر الدخول
                    valid_resistances = [r for r in resistance_levels if r > entry_price]
                    if valid_resistances:
                        strongest_resistance = min(valid_resistances)
                        # جعل وقف الخسارة فوق مستوى المقاومة بقليل
                        sl_price = strongest_resistance * (1 + volatility/2)
                        return min(sl_price, entry_price * (1 + volatility * 3))
                
                # إذا لم نجد مقاومة قوية، نستخدم المتوسط المتحرك أو نسبة ثابتة
                return max(entry_price * (1 + volatility * 2), ma_20 * (1 + volatility))
        
        except Exception as e:
            logging.error(f"Error calculating support/resistance SL: {str(e)}")
            # في حالة الخطأ، نعود إلى الطريقة التقليدية
            if side == "BUY":
                return entry_price * (1 - volatility * 2)
            else:
                return entry_price * (1 + volatility * 2)
    
    def _check_leverage_compatibility(self, symbol, position_size, entry_price, leverage, balance):
        """
        التحقق من توافق الرافعة المالية مع حجم المركز
        """
        # حساب قيمة المركز
        position_value = position_size * entry_price
        
        # حساب نسبة المركز من الرصيد
        position_percentage = position_value / balance * 100
        
        # إذا كانت نسبة المركز صغيرة جداً بالنسبة للرافعة
        if position_percentage < 5 and leverage > 5:
            recommended_leverage = min(5, max(1, int(position_percentage / 5)))
            return False, recommended_leverage
        
        # إذا كانت نسبة المركز كبيرة جداً بالنسبة للرافعة
        max_safe_position = balance * leverage * 0.8  # 80% من الحد الأقصى الآمن
        if position_value > max_safe_position:
            recommended_leverage = min(leverage, max(1, int(position_value / balance * 1.2)))
            return False, recommended_leverage
        
        return True, leverage
            

    async def set_leverage(self, symbol, leverage):
        """ضبط الرافعة المالية للزوج المحدد"""
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                # تنظيف رمز الزوج للتأكد من أنه متوافق مع Binance
                clean_symbol = symbol.replace("/", "").replace("USDT", "USDT")
                
                if TRADING_MODE == "real":
                    # استخدام واجهة CCxt الصحيحة لضبط الرافعة
                    # ملاحظة: قد تختلف طريقة استدعاء set_leverage حسب إصدار ccxt
                    # جرب هذه الطرق المختلفة
                    try:
                        # الطريقة 1: استخدام الاسم الكامل للزوج
                        response = self.exchange.set_leverage(leverage, clean_symbol)
                    except Exception as e:
                        try:
                            # الطريقة 2: استخدام رمز الزوج بدون / 
                            response = self.exchange.set_leverage(leverage, symbol.replace("/", ""))
                        except Exception as e2:
                            # الطريقة 3: استخدام رمز الزوج كما هو
                            response = self.exchange.set_leverage(leverage, symbol)
                    
                    await send_telegram_message_async(f"✅ تم ضبط الرافعة المالية لـ {symbol} إلى {leverage}x")
                    return True
                else:
                    # في الوضع التجريبي، لا حاجة لضبط الرافعة فعلياً
                    await send_telegram_message_async(f"🔵 [تجريبي] تم ضبط الرافعة المالية لـ {symbol} إلى {leverage}x")
                    return True
                    
            except Exception as e:
                error_msg = f"❌ المحاولة {attempt+1} لضبط الرافعة لـ {symbol} فشلت: {str(e)}"
                logging.error(error_msg)
                
                if attempt == max_retries - 1:
                    await send_telegram_message_async(f"❌ فشل جميع المحاولات لضبط الرافعة لـ {symbol}: {str(e)}")
                    return False
                else:
                    # الانتظار قبل إعادة المحاولة
                    wait_time = (attempt + 1) * 2
                    await asyncio.sleep(wait_time)

    async def record_partial_profit(self, trade_id, tp_level, profit_amount, closed_size):
        """تسجيل ربح جزئي عند تحقيق أحد أهداف جني الأرباح"""
        try:
            if trade_id not in self.open_trades:
                return False
                
            trade = self.open_trades[trade_id]
            
            # إضافة الربح الجزئي إلى السجل
            if 'partial_profits' not in trade:
                trade['partial_profits'] = []
                
            trade['partial_profits'].append({
                'timestamp': datetime.now(),
                'tp_level': tp_level,
                'profit': profit_amount,
                'closed_size': closed_size
            })
            
            # زيادة عداد مرات تحقيق الأهداف
            trade['tp_hit_level'] = trade.get('tp_hit_level', 0) + 1
            
            # حفظ الحالة بعد التحديث
            self.save_state()
            
            return True
            
        except Exception as e:
            print(f"❌ خطأ في تسجيل الربح الجزئي: {str(e)}")
            return False

    from datetime import datetime


    async def check_and_open_trade(self, symbol, signal_data, df=None, predictions=None):
        """التحقق من شروط فتح صفقة تلقائية وفتحها إذا توافرت"""
        trade_id = None
        
        try:
            # تحقق أن التداول التلقائي مفعّل
            if not self.auto_trading_enabled:
                msg = f"⚠️ التداول التلقائي غير مفعّل - لن يتم فتح صفقة لـ {symbol}"
                print(msg)
                return None

            # إضافة رمز الزوج إلى بيانات الإشارة إذا لم يكن موجوداً
            if 'symbol' not in signal_data:
                signal_data['symbol'] = symbol

            # تحقق من شروط فتح الصفقة
            if not self.should_open_trade(signal_data):
                msg = f"⚠️ لم يتم فتح صفقة لـ {symbol} - الشروط لم تتحقق"
                print(msg)
                return None

            # الحصول على السعر الحالي
            current_price = await self.get_current_price(symbol)
            if current_price <= 0:
                msg = f"⚠️ لم يتم فتح صفقة لـ {symbol} - سعر غير صالح"
                print(msg)
                return None

            # حساب حجم المركز - سيقوم هذا بحساب stop_loss تلقائياً بناءً على المستويات
            position_size = await self.calculate_position_size(symbol, signal_data)
            if position_size <= 0:
                msg = f"⚠️ لم يتم فتح صفقة لـ {symbol} - حجم غير صالح: {position_size}"
                print(msg)
                return None

            # استخدام وقف الخسارة الذي تم حسابه في calculate_position_size
            stop_loss = signal_data.get('calculated_sl', 0)


            # تحديد اتجاه الصفقة
            side = "BUY" if signal_data.get('overall_direction') == "صعود" else "SELL"

            # استخدام وقف الخسارة المحسوب من التقرير الذكي
            stop_loss = signal_data.get('calculated_sl', 0)
            if stop_loss <= 0:
                # إذا لم يتم حساب وقف الخسارة، استخدام النسبة الثابتة كاحتياطي
                stop_loss_percentage = 0.02
                if side == "BUY":
                    stop_loss = current_price * (1 - stop_loss_percentage)
                else:
                    stop_loss = current_price * (1 + stop_loss_percentage)
                print(f"📊 استخدام وقف خسارة افتراضي: {stop_loss}")

            # الحصول على أهداف جني الأرباح من الإشارة أو إنشاء أهداف افتراضية
            tp_levels = signal_data.get('global_tp', [])
            if not tp_levels:
                if side == "BUY":
                    tp_levels = [
                        current_price * 1.02,
                        current_price * 1.04,
                        current_price * 1.06
                    ]
                else:
                    tp_levels = [
                        current_price * 0.98,
                        current_price * 0.96,
                        current_price * 0.94
                    ]
                print(f"📊 استخدام أهداف افتراضية: {tp_levels}")

            # فتح الصفقة باستخدام await مع تمرير التوقعات
            trade_id = await self.place_trade_order(
                symbol,
                side,
                position_size,
                tp_levels,
                stop_loss,
                entry_price=current_price,
                df=df,
                predictions=predictions
            )


            if trade_id:
                # تسجيل الصفقة المفتوحة مع الحقول الجديدة
                self.open_trades[trade_id] = {
                    'symbol': symbol,
                    'side': side,
                    'entry_price': current_price,
                    'tp_levels': tp_levels,
                    'sl_level': stop_loss,
                    'size': position_size,
                    'timestamp': datetime.now(),
                    'signal_confidence': signal_data.get('avg_confidence', 0),
                    'leverage': self.leverage,
                    'partial_profits': [],  # لتسجيل الأرباح الجزئية
                    'tp_hit_level': 0,      # عدد مرات تحقيق أهداف جني الأرباح
                    'open_reason': 'إشارة تداول تلقائية'
                }


                # حفظ الحالة بعد فتح الصفقة
                self.save_state()

                # وضع أوامر وقف الخسارة وجني الأرباح
                self.place_tp_sl_orders(
                    symbol, side, position_size,
                    tp_levels, stop_loss,
                    entry_price=current_price, df=df, predictions=predictions
                )

                   # إرسال رسالة التأكيد باستخدام await
                message = f"✅ تم فتح صفقة تلقائية لـ {symbol} برقم: {trade_id}"
                await send_telegram_message_async(message)
                return trade_id
            else:
                msg = f"⚠️ لم يتم فتح صفقة لـ {symbol} - أمر السوق فشل"
                print(msg)
                return None

        except Exception as e:
            msg = f"❌ خطأ في فتح الصفقة لـ {symbol}: {str(e)}"
            print(msg)
            return None


    
    async def place_trade_order(self, symbol, side, size, tp_levels, sl_level, entry_price=None, df=None, predictions=None):
        """فتح صفقة مع أوامر وقف الخسارة وجني الأرباح"""
        try:
            if TRADING_MODE == "demo":
                trade_id = f"DEMO_{int(time.time())}_{symbol}"
                await self.notify_trade_opened(symbol, side, size, tp_levels, sl_level, 0.7)
                return trade_id

            clean_symbol = symbol.replace("/", "")

            # استخدام await مع create_order
            order = await self.exchange.create_order(clean_symbol, 'market', side, size)
            print("🔹 رد باينانس:", order)

            if order and order.get('id'):
                trade_id = order['id']
                
                # تحديد مستويات الدعم والمقاومة من التوقعات
                key_levels = identify_key_levels(predictions) if predictions else {}
                
                # استخدام await مع وضع أوامر TP/SL مع المستويات المحددة
                await self.place_tp_sl_orders(
                    clean_symbol,
                    side,
                    size,
                    tp_levels,
                    sl_level,
                    entry_price,
                    df,
                    predictions,
                    key_levels
                )
                
                await self.notify_trade_opened(symbol, side, size, tp_levels, sl_level, 0.7)
                
                return trade_id
            else:
                print(f"⚠️ لم يتم فتح صفقة لـ {symbol} – أمر السوق فشل أو لم يُرجع معرف")
                return None

        except Exception as e:
            print(f"❌ فشل فتح صفقة {symbol}: {str(e)}")
            return None



    def should_open_trade(self, signal_data):
        """التحقق من شروط فتح الصفقة مع تحسين المرونة"""
        # التحقق من وجود بيانات الإشارة الأساسية
        if not signal_data:
            print("⚠️ لا توجد بيانات إشارة")
            return False

        # التحقق من الثقة الدنيا مع مرونة أكثر
        min_confidence = self.min_confidence
        confidence = signal_data.get('avg_confidence', 0)
        
        # إذا كانت هناك إشارات قوية أخرى، يمكن تخفيض حد الثقة
        has_strong_trend = signal_data.get('strong_trend', False)
        has_multi_tf_confirmed = signal_data.get('multi_tf_confirmed', False)
        
        if has_strong_trend and has_multi_tf_confirmed:
            min_confidence = max(0.5, min_confidence * 0.7)  # تخفيض حد الثقة بنسبة 30%
            print(f"📊 تم تخفيض حد الثقة إلى {min_confidence} بسبب إشارات قوية")
        
        if confidence < min_confidence:
            print(f"⚠️ الثقة منخفضة: {confidence:.3f} < {min_confidence:.3f}")
            return False

        # التحقق من وجود اتجاه واضح
        if signal_data.get('overall_direction') not in ["صعود", "هبوط"]:
            print("⚠️ لا يوجد اتجاه واضح")
            return False
        
        # التحقق من وجود أهداف ووقف خسارة
        if not signal_data.get('global_tp') or not signal_data.get('global_sl'):
            print("⚠️ لا توجد أهداف أو وقف خسارة")
            return False
        
        # منع التكرار - التحقق من عدم وجود صفقة مفتوحة حديثاً على نفس الزوج
        current_time = datetime.now()
        symbol = signal_data.get('symbol', '')
        
        # مزامنة فورية مع Binance قبل التحقق
        asyncio.create_task(self.sync_with_binance(symbol))
        
        for trade_id, trade in list(self.open_trades.items()):
            if trade['symbol'] == symbol:
                # إذا كانت الصفقة قديمة (أكثر من 48 ساعة)، اعتبارها مغلقة
                time_diff = (current_time - trade['timestamp']).total_seconds()
                if time_diff > 172800:  # 48 ساعة
                    print(f"🔄 إزالة صفقة قديمة: {trade_id} (منذ {time_diff/3600:.1f} ساعة)")
                    del self.open_trades[trade_id]
                    continue
                    
                # إذا كانت الصفقة حديثة (أقل من 2 ساعة)، منع فتح صفقة جديدة
                if time_diff < 7200:  # 2 ساعة
                    print(f"⚠️ توجد صفقة حديثة على نفس الزوج: {trade_id} (منذ {time_diff/60:.1f} دقيقة)")
                    return False
        
        print(f"✅ جميع شروط فتح الصفقة متوفرة للزوج {symbol}")
        return True





    def calculate_optimal_tp(self, side, entry_price, support_levels, resistance_levels, 
                             default_tp_levels, volatility, confidence):
        """حساب أهداف جني الأرباح الأمثلة بناءً على متعددة معايير"""
        
        if side == "BUY":  # صفقة شراء
            # استخدام مستويات المقاومة كأهداف
            valid_resistances = [r for r in resistance_levels if r > entry_price]
            
            if not valid_resistances:
                return default_tp_levels
            
            # ترتيب مستويات المقاومة حسب القوة والقرب
            valid_resistances.sort()
            
            # اختيار أقوى 3-5 مستويات كأهداف
            optimal_targets = valid_resistances[:min(5, len(valid_resistances))]
            
            # تعديل الأهداف بناءً على الثقة والتقلبات
            adjusted_targets = []
            for target in optimal_targets:
                # حساب نسبة الربح المستهدفة
                profit_pct = (target - entry_price) / entry_price
                
                # تعديل الهدف بناءً على الثقة
                if confidence > 0.7:
                    target *= 1.05  # زيادة الهدف بنسبة 5% إذا كانت الثقة عالية
                elif confidence < 0.5:
                    target *= 0.95  # تقليل الهدف بنسبة 5% إذا كانت الثقة منخفضة
                
                # تعديل الهدف بناءً على التقلبات
                if volatility > 0.08:  # تقلبات عالية
                    target *= 1.08  # زيادة الهدف لتعويض المخاطرة
                elif volatility < 0.03:  # تقلبات منخفضة
                    target *= 0.95  # تقليل الهدف بسبب محدودية الحركة
                
                adjusted_targets.append(target)
            
            return adjusted_targets[:len(default_tp_levels)]  # الحفاظ على عدد الأهداف الأصلي
            
        else:  # صفقة بيع
            # استخدام مستويات الدعم كأهداف
            valid_supports = [s for s in support_levels if s < entry_price]
            
            if not valid_supports:
                return default_tp_levels
            
            valid_supports.sort(reverse=True)  # ترتيب تنازلي
            
            optimal_targets = valid_supports[:min(5, len(valid_supports))]
            
            adjusted_targets = []
            for target in optimal_targets:
                profit_pct = (entry_price - target) / entry_price
                
                if confidence > 0.7:
                    target *= 0.95  # للبيع: تقليل الهدف (لأننا نريد سعر أقل)
                elif confidence < 0.5:
                    target *= 1.05  # للبيع: زيادة الهدف
                
                if volatility > 0.08:
                    target *= 0.92  # للبيع: تقليل الهدف أكثر
                elif volatility < 0.03:
                    target *= 1.05  # للبيع: زيادة الهدف
                
                adjusted_targets.append(target)
            
            return adjusted_targets[:len(default_tp_levels)]


    def calculate_optimal_sl(self, side, entry_price, support_levels, resistance_levels, 
                             market_trend, volatility, confidence):
        """حساب وقف الخسارة الأمثل بناءً على متعددة معايير"""
        
        # القيمة الافتراضية (الاحتياطية)
        default_sl = entry_price * 0.97 if side == "BUY" else entry_price * 1.03
        
        if side == "BUY":  # صفقة شراء
            # استخدام أقوى مستوى دعم تحت سعر الدخول
            valid_supports = [s for s in support_levels if s < entry_price]
            
            if not valid_supports:
                return default_sl
            
            # ترجيح المستويات بناءً على قوة الدعم والتقلبات والثقة
            weighted_levels = []
            for support in valid_supports:
                # حساب المسافة من سعر الدخول
                distance_pct = (entry_price - support) / entry_price
                
                # وزن المستوى بناءً على قوته وملاءمته للظروف الحالية
                weight = 1.0
                
                # زيادة الوزن إذا كان الاتجاه العام صعودي
                if market_trend == "صعود":
                    weight *= 1.2
                
                # تعديل الوزن بناءً على التقلبات
                if volatility > 0.05:  # تقلبات عالية
                    weight *= 0.8  # تقليل الوزن للمستويات البعيدة
                else:
                    weight *= 1.2  # زيادة الوزن للمستويات القريبة
                
                # تعديل الوزن بناءً على الثقة
                weight *= confidence
                
                weighted_levels.append((support, weight))
            
            # اختيار المستوى بأعلى وزن
            if weighted_levels:
                weighted_levels.sort(key=lambda x: x[1], reverse=True)
                return weighted_levels[0][0]
            else:
                return default_sl
                
        else:  # صفقة بيع
            # استخدام أقوى مستوى مقاومة فوق سعر الدخول
            valid_resistances = [r for r in resistance_levels if r > entry_price]
            
            if not valid_resistances:
                return default_sl
            
            weighted_levels = []
            for resistance in valid_resistances:
                # حساب المسافة من سعر الدخول
                distance_pct = (resistance - entry_price) / entry_price
                
                weight = 1.0
                
                if market_trend == "هبوط":
                    weight *= 1.2
                
                if volatility > 0.05:
                    weight *= 0.8
                else:
                    weight *= 1.2
                
                weight *= confidence
                
                weighted_levels.append((resistance, weight))
            
            if weighted_levels:
                weighted_levels.sort(key=lambda x: x[1], reverse=True)
                return weighted_levels[0][0]
            else:
                return default_sl

    
    async def calculate_position_size(self, symbol, signal_data, use_leverage=True):
        """حجم المركز بناء على إدارة المخاطر مع مراعاة مستويات الدعم والمقاومة"""
        try:
            # الحصول على الرصيد المتاح
            balance = await self.get_available_balance()
            if balance <= 0:
                print(f"⚠️ الرصيد المتاح صفر أو سالب: {balance}")
                return 0

            # الحصول على السعر الحالي
            current_price = await self.get_current_price(symbol)
            if current_price <= 0:
                print(f"⚠️ السعر الحالي غير صالح: {current_price}")
                return 0

            # تحديد اتجاه الصفقة
            side = "BUY" if signal_data.get('overall_direction') == "صعود" else "SELL"

            # استخدام مستويات الدعم والمقاومة من التقرير الذكي
            key_levels = signal_data.get('analysis', {}).get('key_levels', {})

            # حساب وقف الخسارة بناءً على مستويات الدعم/المقاومة
            stop_loss = calculate_smart_stop_loss(side, current_price, key_levels)

            print(f"📊 استخدام وقف خسارة ذكي: {stop_loss}")

            # حساب مسافة الوقف
            if side == "BUY":
                risk_distance = abs(current_price - stop_loss)
            else:
                risk_distance = abs(stop_loss - current_price)

            print(f"🔧 حساب الحجم: السعر={current_price}, الوقف={stop_loss}, المسافة={risk_distance}")

            if risk_distance <= 0:
                print(f"⚠️ مسافة الوقف غير صحيحة: {risk_distance}")
                return 0

            # حساب المخاطرة بالدولار
            risk_amount = balance * self.risk_per_trade

            # تطبيق الرافعة المالية إذا كانت مفعلة
            leverage_factor = self.leverage if use_leverage else 1
            risk_amount *= leverage_factor

            # حساب حجم المركز
            position_size = risk_amount / risk_distance

            # التحقق من الحد الأدنى والأقصى للحجم
            min_size = await self.get_min_order_size(symbol)
            max_size = await self.get_max_order_size(symbol, balance)

            # وضع حد أدنى افتراضي إذا كان صغير جداً
            min_trade_size = min_size
            final_size = max(min_trade_size, min(position_size, max_size))

            print(f"📊 حجم المركز النهائي: {final_size} (من {position_size})")

            # تخزين وقف الخسارة المحسوب في signal_data لاستخدامه لاحقاً
            signal_data['calculated_sl'] = stop_loss

            return final_size

        except Exception as e:
            print(f"❌ خطأ في حساب حجم المركز: {str(e)}")
            return 0





    
    async def handle_stop_loss(self, trade_id, current_price):
        """معالجة وقف الخسارة مع الأخذ في الاعتبار الأرباح الجزئية المحققة"""
        try:
            if trade_id not in self.open_trades:
                return
                
            trade = self.open_trades[trade_id]
            
            # حساب إجمالي الربح/الخسارة مع الأرباح الجزئية
            total_profit = 0
            
            # جمع الأرباح الجزئية المحققة مسبقاً
            if 'partial_profits' in trade:
                for partial in trade['partial_profits']:
                    total_profit += partial['profit']
            
            # حساب ربح/خسارة الجزء المتبقي من الصفقة
            remaining_size = trade['size']
            
            # طرح الحجم الذي تم إغلاقه مسبقاً في الأرباح الجزئية
            if 'partial_profits' in trade:
                for partial in trade['partial_profits']:
                    remaining_size -= partial.get('closed_size', 0)
            
            if remaining_size > 0:
                if trade['side'] == 'BUY':
                    profit_pct = ((current_price - trade['entry_price']) / trade['entry_price']) * 100
                else:
                    profit_pct = ((trade['entry_price'] - current_price) / trade['entry_price']) * 100
                
                gross_profit = remaining_size * profit_pct / 100
                fee = gross_profit * self.trading_fee_rate
                net_profit = gross_profit - fee
                total_profit += net_profit
            
            # إعداد رسالة وقف الخسارة
            message = (
                f"🛑 <b>تم تفعيل وقف الخسارة لصفقة {trade['symbol']}</b>\n"
                f"📊 الاتجاه: {'شراء 🔺' if trade['side'] == 'BUY' else 'بيع 🔻'}\n"
                f"💰 السعر الحالي: {current_price:.6f}\n"
                f"🛑 وقف الخسارة: {trade['sl_level']:.6f}\n"
                f"📉 الربح/الخسارة النهائية: {total_profit:+.2f} USDT\n"
            )
            
            if trade.get('tp_hit_level', 0) > 0:
                message += f"🎯 الأهداف المحققة: {trade['tp_hit_level']} من {len(trade['tp_levels'])}"
            
            # إرسال الرسالة
            await send_telegram_message_async(message)
            
            # إغلاق الصفقة
            await self.close_trade(trade_id, "وقف خسارة")
            
        except Exception as e:
            print(f"❌ خطأ في معالجة وقف الخسارة: {str(e)}")
            # في حالة الخطأ، نغلق الصفقة دون حساب الأرباح
            await self.close_trade(trade_id, "وقف خسارة (مع خطأ في الحساب)")
    
    async def place_tp_sl_orders(self, symbol, side, size, tp_levels, sl_level, entry_price=None,
                                 df=None, predictions=None, key_levels=None):
        """وضع أوامر جني الأرباح ووقف الخسارة بناءً على مستويات الدعم والمقاومة"""
        try:
            clean_symbol = symbol.replace("/", "")

            # تحديد الجانب المعاكس لأوامر الإغلاق
            close_side = "SELL" if side == "BUY" else "BUY"

            # استخدام مستويات الدعم والمقاومة إذا كانت متاحة
            if key_levels:
                if side == "BUY":
                    # لصفقات الشراء: وضع وقف الخسارة عند أقوى مستوى دعم
                    if key_levels.get('strong_support'):
                        sl_level = min(key_levels['strong_support'])  # أخذ أقوى مستوى دعم (أدنى سعر)
                        print(f"📊 استخدام أقوى مستوى دعم لوقف الخسارة: {sl_level}")
                else:
                    # لصفقات البيع: وضع وقف الخسارة عند أقوى مستوى مقاومة
                    if key_levels.get('strong_resistance'):
                        sl_level = max(key_levels['strong_resistance'])  # أخذ أقوى مستوى مقاومة (أعلى سعر)
                        print(f"📊 استخدام أقوى مستوى مقاومة لوقف الخسارة: {sl_level}")

            # وضع أمر وقف الخسارة (Stop Loss)
            try:
                sl_order = await self.exchange.create_order(
                    clean_symbol,
                    'STOP_MARKET',
                    close_side,
                    size,
                    None,
                    {
                        'stopPrice': sl_level,
                        'reduceOnly': True,
                        'workingType': 'MARK_PRICE'
                    }
                )
                print(f"✅ تم وضع أمر وقف الخسارة عند: {sl_level}")
            except Exception as sl_error:
                print(f"❌ خطأ في وضع وقف الخسارة: {sl_error}")
                # حاول مرة أخرى بنوع مختلف إذا لزم الأمر
                try:
                    sl_order = await self.exchange.create_order(
                        clean_symbol,
                        'STOP',
                        close_side,
                        size,
                        None,
                        {
                            'stopPrice': sl_level,
                            'reduceOnly': True
                        }
                    )
                    print(f"✅ تم وضع أمر وقف الخسارة (نوع STOP) عند: {sl_level}")
                except Exception as sl_error2:
                    print(f"❌ فشل وضع وقف الخسارة تماماً: {sl_error2}")

            # وضع أوامر جني الأرباح (Take Profit)
            for i, tp_level in enumerate(tp_levels, 1):
                try:
                    # حساب حجم كل هدف بشكل منفصل
                    tp_size = size / len(tp_levels)

                    tp_order = await self.exchange.create_order(
                        clean_symbol,
                        'LIMIT',
                        close_side,
                        tp_size,
                        tp_level,
                        {
                            'reduceOnly': True,
                            'timeInForce': 'GTC'
                        }
                    )
                    print(f"✅ تم وضع أمر جني الأرباح {i} عند: {tp_level}")
                except Exception as tp_error:
                    print(f"❌ خطأ في وضع أمر جني الأرباح {i}: {tp_error}")
                    # حاول استخدام أمر TAKE_PROFIT إذا كان متاحاً
                    try:
                        tp_order = await self.exchange.create_order(
                            clean_symbol,
                            'TAKE_PROFIT',
                            close_side,
                            tp_size,
                            None,
                            {
                                'stopPrice': tp_level,
                                'reduceOnly': True
                            }
                        )
                        print(f"✅ تم وضع أمر جني الأرباح {i} (نوع TAKE_PROFIT) عند: {tp_level}")
                    except Exception as tp_error2:
                        print(f"❌ فشل وضع أمر جني الأرباح {i} تماماً: {tp_error2}")

        except Exception as e:
            print(f"❌ خطأ عام في وضع أوامر TP/SL: {str(e)}")


    def find_strong_support_resistance(self, df, lookback_period=50, num_levels=5):
        """
        اكتشاف أقوى نقاط الدعم والمقاومة من البيانات التاريخية
        """
        try:
            # التحقق من أن df ليس None وأنه يحتوي على بيانات
            if df is None or df.empty:
                print("⚠️ بيانات فارغة أو غير متوفرة للتحليل")
                return [], []
            
            # التأكد من أن هناك بيانات كافية للتحليل
            if len(df) < lookback_period * 2:
                print(f"⚠️ لا توجد بيانات كافية للتحليل (مطلوب: {lookback_period * 2}، متوفر: {len(df)})")
                return [], []
            
            # استخدام أعلى وأقل الأسعار لفترة Lookback
            high_points = []
            low_points = []
            
            # تحليل البيانات لاكتشاف القمم والقيعان
            for i in range(lookback_period, len(df) - lookback_period):
                # التحقق من القمم (المقاومة)
                if df['high'].iloc[i] == df['high'].iloc[i-lookback_period:i+lookback_period].max():
                    high_points.append({
                        'price': df['high'].iloc[i],
                        'strength': self.calculate_level_strength(df, i, 'resistance')
                    })
                
                # التحقق من القيعان (الدعم)
                if df['low'].iloc[i] == df['low'].iloc[i-lookback_period:i+lookback_period].min():
                    low_points.append({
                        'price': df['low'].iloc[i],
                        'strength': self.calculate_level_strength(df, i, 'support')
                    })
            
            # ترتيب النقاط حسب القوة
            high_points.sort(key=lambda x: x['strength'], reverse=True)
            low_points.sort(key=lambda x: x['strength'], reverse=True)
            
            # أخذ أقوى النقاط
            strongest_resistance = [p['price'] for p in high_points[:num_levels]]
            strongest_support = [p['price'] for p in low_points[:num_levels]]
            
            print(f"📊 تم العثور على {len(strongest_support)} نقطة دعم و {len(strongest_resistance)} نقطة مقاومة")
            
            return strongest_support, strongest_resistance
            
        except Exception as e:
            print(f"❌ خطأ في إيجاد نقاط الدعم والمقاومة: {str(e)}")
            return [], []



    def calculate_level_strength(self, df, index, level_type):
        """
        حساب قوة مستوى الدعم أو المقاومة
        """
        try:
            strength = 0
            current_price = df['close'].iloc[index]
            
            # عدد المرات التي تم اختبار المستوى فيها
            test_count = 0
            
            # حجم التداول عند المستوى
            volume_at_level = 0
            
            # المدة منذ آخر اختبار
            recency_factor = 1
            
            # البحث في نطاق 100 شمعة حول النقطة
            for i in range(max(0, index-100), min(len(df), index+100)):
                if level_type == 'support':
                    if abs(df['low'].iloc[i] - current_price) / current_price < 0.005:  # 0.5% tolerance
                        test_count += 1
                        volume_at_level += df['volume'].iloc[i]
                else:  # resistance
                    if abs(df['high'].iloc[i] - current_price) / current_price < 0.005:  # 0.5% tolerance
                        test_count += 1
                        volume_at_level += df['volume'].iloc[i]
            
            # عامل الحداثة (كلما كان أقرب للوقت الحالي، كلما كان أقوى)
            recency_factor = 1 - (len(df) - index) / len(df)
            
            # حساب القوة النهائية
            strength = test_count * 0.4 + (volume_at_level * 0.0001) * 0.3 + recency_factor * 0.3
            
            return strength
            
        except Exception as e:
            print(f"❌ خطأ في حساب قوة المستوى: {str(e)}")
            return 0.5  # قيمة افتراضية


    def calculate_optimal_sl_based_on_sr(self, side, entry_price, support_levels, resistance_levels):
        """
        حساب وقف الخسارة الأمثل بناءً على نقاط الدعم والمقاومة
        """
        try:
            if side == "BUY":
                # لصفقات الشراء: استخدام أقوى مستوى دعم تحت سعر الدخول
                valid_supports = [s for s in support_levels if s < entry_price]
                if valid_supports:
                    # أخذ أقوى مستوى دعم (أعلى سعر من مستويات الدعم)
                    strongest_support = max(valid_supports)
                    # وضع وقف الخسارة تحت مستوى الدعم بقليل (1%)
                    return strongest_support * 0.99
                else:
                    # إذا لم يوجد دعم، استخدام نسبة ثابتة (2%)
                    return entry_price * 0.98
            else:
                # لصفقات البيع: استخدام أقوى مستوى مقاومة فوق سعر الدخول
                valid_resistances = [r for r in resistance_levels if r > entry_price]
                if valid_resistances:
                    # أخذ أقوى مستوى مقاومة (أقل سعر من مستويات المقاومة)
                    strongest_resistance = min(valid_resistances)
                    # وضع وقف الخسارة فوق مستوى المقاومة بقليل (1%)
                    return strongest_resistance * 1.01
                else:
                    # إذا لم يوجد مقاومة، استخدام نسبة ثابتة (2%)
                    return entry_price * 1.02
                    
        except Exception as e:
            print(f"❌ خطأ في حساب وقف الخسارة: {str(e)}")
            # استخدام نسبة ثابتة في حالة الخطأ
            return entry_price * (0.98 if side == "BUY" else 1.02)

    def calculate_optimal_tp_based_on_sr(self, side, entry_price, support_levels, resistance_levels, default_tp_levels):
        """
        حساب أهداف جني الأرباح الأمثلة بناءً على نقاط الدعم والمقاومة
        """
        try:
            if side == "BUY":
                # لصفقات الشراء: استخدام مستويات المقاومة كأهداف
                valid_resistances = [r for r in resistance_levels if r > entry_price]
                if valid_resistances:
                    # ترتيب مستويات المقاومة تصاعدياً وأخذ أقرب 3 مستويات
                    valid_resistances.sort()
                    optimal_targets = valid_resistances[:min(3, len(valid_resistances))]
                    return optimal_targets
                else:
                    # استخدام الأهداف الافتراضية إذا لم توجد مقاومة
                    return default_tp_levels if default_tp_levels else [entry_price * 1.02, entry_price * 1.04, entry_price * 1.06]
            else:
                # لصفقات البيع: استخدام مستويات الدعم كأهداف
                valid_supports = [s for s in support_levels if s < entry_price]
                if valid_supports:
                    # ترتيب مستويات الدعم تنازلياً وأخذ أقرب 3 مستويات
                    valid_supports.sort(reverse=True)
                    optimal_targets = valid_supports[:min(3, len(valid_supports))]
                    return optimal_targets
                else:
                    # استخدام الأهداف الافتراضية إذا لم يوجد دعم
                    return default_tp_levels if default_tp_levels else [entry_price * 0.98, entry_price * 0.96, entry_price * 0.94]
                    
        except Exception as e:
            print(f"❌ خطأ في حساب أهداف جني الأرباح: {str(e)}")
            # استخدام الأهداف الافتراضية في حالة الخطأ
            return default_tp_levels if default_tp_levels else (
                [entry_price * 1.02, entry_price * 1.04, entry_price * 1.06] if side == "BUY" 
                else [entry_price * 0.98, entry_price * 0.96, entry_price * 0.94]
            )

    async def send_trading_decision_report(self, symbol, side, entry_price, sl_level, tp_levels, support_levels, resistance_levels):
        """
        إرسال تقرير مفصل عن قرارات التداول
        """
        try:
            message = f"""
    📊 **تقرير قرارات التداول - {symbol}**

    • **الاتجاه:** {'شراء 🔺' if side == 'BUY' else 'بيع 🔻'}
    • **سعر الدخول:** {entry_price:.6f}
    • **وقف الخسارة:** {sl_level:.6f} (مسافة: {abs(entry_price - sl_level)/entry_price*100:.2f}%)

    🎯 **أهداف جني الأرباح:**
    """
            
            for i, tp in enumerate(tp_levels, 1):
                profit_pct = ((tp - entry_price) / entry_price * 100) if side == "BUY" else ((entry_price - tp) / entry_price * 100)
                message += f"       {i}. {tp:.6f} ({profit_pct:+.2f}%)\n"
            
            if support_levels:
                message += f"""
    📈 **نقاط الدعم القوية:**
    """
                for i, support in enumerate(support_levels[:3], 1):
                    message += f"       {i}. {support:.6f}\n"
            
            if resistance_levels:
                message += f"""
    📉 **نقاط المقاومة القوية:**
    """
                for i, resistance in enumerate(resistance_levels[:3], 1):
                    message += f"       {i}. {resistance:.6f}\n"
            
            message += f"""
    ⏰ **وقت وضع الأوامر:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    """
            
            await send_telegram_message_async(message)
            
        except Exception as e:
            print(f"❌ خطأ في إرسال تقرير التداول: {str(e)}")


    
    async def get_current_price(self, symbol):
        """الحصول على السعر الحالي للزوج"""
        try:
            # تنظيف رمز العملة أولاً
            clean_symbol = symbol.replace("/", "").replace("USDT", "USDT")
            
            # استخدام fetch_ticker للحصول على السعر
            ticker = await self.exchange.fetch_ticker(clean_symbol)
            
            if ticker and 'last' in ticker and ticker['last']:
                return float(ticker['last'])
            elif ticker and 'close' in ticker and ticker['close']:
                return float(ticker['close'])
            else:
                logger.error(f"لم يتم العثور على سعر صالح لـ {symbol}")
                return 0
        except Exception as e:
            logger.error(f"خطأ في جلب السعر الحالي لـ {symbol}: {str(e)}")
            return 0

    async def send_tp_notification(self, trade, tp_level, tp_number, current_price):
        """إرسال إشعار تحقيق هدف ربح"""
        try:
            message = (
                f"🎯 <b>تم تحقيق هدف الربح #{tp_number}</b>\n"
                f"• الزوج: {trade['symbol']}\n"
                f"• الاتجاه: {'شراء 🔺' if trade['side'] == 'BUY' else 'بيع 🔻'}\n"
                f"• سعر الدخول: {trade['entry_price']:.6f}\n"
                f"• الهدف: {tp_level:.6f}\n"
                f"• السعر الحالي: {current_price:.6f}\n"
                f"• الحجم: {trade['size']:.6f}\n"
                f"• الرافعة: {trade.get('leverage', 1)}x\n"
                f"• الوقت: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            
            await send_telegram_message_async(message)
            
        except Exception as e:
            print(f"❌ خطأ في إرسال إشعار تحقيق الهدف: {str(e)}")

    async def send_sl_notification(self, trade, current_price):
        """إرسال إشعار وقف الخسارة"""
        try:
            # حساب الربح/الخسارة
            pnl_data = await self.calculate_trade_pnl(trade, current_price)
            
            message = (
                f"🛑 <b>تم تنفيذ وقف الخسارة</b>\n"
                f"• الزوج: {trade['symbol']}\n"
                f"• الاتجاه: {'شراء 🔺' if trade['side'] == 'BUY' else 'بيع 🔻'}\n"
                f"• سعر الدخول: {trade['entry_price']:.6f}\n"
                f"• وقف الخسارة: {trade['sl_level']:.6f}\n"
                f"• السعر الحالي: {current_price:.6f}\n"
                f"• الخسارة: {pnl_data['net']:.6f} USDT\n"
                f"• النسبة: {pnl_data['roe']:.2f}%\n"
                f"• الحجم: {trade['size']:.6f}\n"
                f"• الوقت: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            
            await send_telegram_message_async(message)
            
        except Exception as e:
            print(f"❌ خطأ في إرسال إشعار وقف الخسارة: {str(e)}")

    async def send_periodic_report(self, active_trades, profit_trades, loss_trades):
        """إرسال تقرير دوري عن الصفقات"""
        try:
            if not active_trades:
                return
                
            total_profit = sum(trade['pnl'] for trade in profit_trades)
            total_loss = sum(trade['pnl'] for trade in loss_trades)
            net_pnl = total_profit + total_loss  # total_loss قيمة سالبة
            
            message = (
                f"📊 <b>تقرير التداول الدوري</b>\n"
                f"• عدد الصفقات النشطة: {len(active_trades)}\n"
                f"• الأهداف المحققة: {len(profit_trades)}\n"
                f"• أوامر الوقف المنفذة: {len(loss_trades)}\n"
                f"• صافي الربح/الخسارة: {net_pnl:.6f} USDT\n"
                f"• إجمالي الأرباح: {total_profit:.6f} USDT\n"
                f"• إجمالي الخسائر: {total_loss:.6f} USDT\n"
                f"• الوقت: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                f"<b>الصفقات النشطة:</b>\n"
            )
            
            # تفاصيل الصفقات النشطة
            for i, trade in enumerate(active_trades, 1):
                message += (
                    f"{i}. {trade['symbol']} | "
                    f"{'شراء 🔺' if trade['side'] == 'BUY' else 'بيع 🔻'} | "
                    f"الدخول: {trade['entry_price']:.6f} | "
                    f"الحالي: {trade['current_price']:.6f} | "
                    f"الأرباح: {trade['pnl']:.6f} USDT\n"
                )
            
            await send_telegram_message_async(message)
            
        except Exception as e:
            print(f"❌ خطأ في إرسال التقرير الدوري: {str(e)}")


    async def get_detailed_trade_info(self, symbol=None):
        """
        الحصول على معلومات مفصلة عن الصفقات من Binance
        """
        try:
            if symbol:
                symbols_to_check = [symbol]
            else:
                symbols_to_check = list(set(trade['symbol'] for trade in self.open_trades.values()))
            
            detailed_info = []
            
            for symbol in symbols_to_check:
                clean_symbol = symbol.replace("/", "")
                
                # الحصول على أوامر مفتوحة من Binance
                open_orders = await self.exchange.fetch_open_orders(clean_symbol)
                
                # الحصول على تاريخ التداولات الأخيرة
                since = int((datetime.now() - timedelta(hours=24)).timestamp() * 1000)
                trades_history = await self.exchange.fetch_my_trades(clean_symbol, since=since)
                
                # جمع المعلومات
                symbol_info = {
                    'symbol': symbol,
                    'open_orders': len(open_orders),
                    'recent_trades': len(trades_history),
                    'orders': []
                }
                
                for order in open_orders:
                    order_info = {
                        'id': order['id'],
                        'side': order['side'],
                        'price': order['price'],
                        'amount': order['amount'],
                        'filled': order['filled'],
                        'status': order['status'],
                        'type': order['type']
                    }
                    symbol_info['orders'].append(order_info)
                
                detailed_info.append(symbol_info)
            
            return detailed_info
            
        except Exception as e:
            print(f"❌ خطأ في الحصول على معلومات مفصلة: {str(e)}")
            return []
    
    async def close_trade(self, trade_id, reason="manual"):
        """إغلاق صفقة وإزالتها من القائمة مع تسجيل سبب الإغلاق"""
        try:
            if trade_id in self.open_trades:
                trade = self.open_trades[trade_id]
                
                # تسجيل سبب الإغلاق
                trade['close_reason'] = reason
                trade['close_time'] = datetime.now()
                
                # حفظ بيانات الصفقة المغلقة للسجلات
                await self.save_closed_trade(trade_id, trade)
                
                # إزالة الصفقة من القائمة النشطة
                del self.open_trades[trade_id]
                
                # حفظ الحالة بعد الإغلاق
                self.save_state()
                
                print(f"✅ تم إغلاق الصفقة {trade_id} بسبب: {reason}")
                return True
            else:
                print(f"⚠️ الصفقة {trade_id} غير موجودة")
                return False
                
        except Exception as e:
            print(f"❌ خطأ في إغلاق الصفقة {trade_id}: {str(e)}")
            return False

    async def save_closed_trade(self, trade_id, trade_data):
        """حفظ بيانات الصفقة المغلقة للسجلات"""
        try:
            # إنشاء مجلد السجلات إذا لم يكن موجوداً
            os.makedirs("trade_history", exist_ok=True)
            
            # اسم الملف based on التاريخ
            filename = f"trade_history/closed_trades_{datetime.now().strftime('%Y-%m')}.json"
            
            # تحميل البيانات الحالية إذا كانت موجودة
            closed_trades = []
            if os.path.exists(filename):
                with open(filename, 'r', encoding='utf-8') as f:
                    closed_trades = json.load(f)
            
            # إضافة الصفقة المغلقة
            trade_data['trade_id'] = trade_id
            closed_trades.append(trade_data)
            
            # حفظ البيانات
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(closed_trades, f, indent=4, ensure_ascii=False)
                
        except Exception as e:
            print(f"❌ خطأ في حفظ سجل الصفقة المغلقة: {str(e)}")


    async def notify_trade_opened(self, symbol, side, size, tp_levels, sl_level, confidence):
        """إرسال إشعار بفتح الصفقة"""
        try:
            message = (
                f"✅ **تم فتح صفقة تلقائية**\n"
                f"• الزوج: {symbol}\n"
                f"• الاتجاه: {'شراء 🔺' if side == 'BUY' else 'بيع 🔻'}\n"
                f"• الحجم: {size:.6f}\n"
                f"• الثقة: {confidence:.1%}\n"
                f"• أهداف الربح (6 مراحل):\n"
            )
            
            for i, tp in enumerate(tp_levels, 1):
                message += f"  {i}. {tp:.6f}\n"
                
            message += (
                f"• وقف الخسارة: {sl_level:.6f}\n"
                f"• الوقت: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            )
            
            # استخدام await مع إرسال الرسالة
            await send_telegram_message_async(message)
            
        except Exception as e:
            print(f"❌ خطأ في إرسال إشعار فتح الصفقة: {str(e)}")


# ====== تهيئة مدير التداول التلقائي ======
auto_trade_manager = AutoTradeManager(exchange, min_confidence=0.7, risk_per_trade=RISK_PER_TRADE)


class BinanceSyncManager:
    def __init__(self, api_key, api_secret):
        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {'defaultType': 'future'},
        })
        self.open_trades = {}
        
    async def sync_open_orders(self):
        """مزامنة الأوامر المفتوحة مع Binance"""
        try:
            # جلب جميع الأوامر النشطة من Binance
            open_orders = await self.exchange.fetch_open_orders()
            
            # تحديث الصفقات المحلية بناءً على حالة البورصة
            for order in open_orders:
                order_id = order['id']
                if order_id in self.open_trades:
                    # تحديث حالة الصفقة بناءً على البيانات من البورصة
                    self.open_trades[order_id]['status'] = order['status']
                    self.open_trades[order_id]['filled'] = order['filled']
                    self.open_trades[order_id]['remaining'] = order['remaining']
            
            # التعرف على الصفقات المغلقة يدوياً
            await self.detect_manually_closed_trades()
            
        except Exception as e:
            print(f"خطأ في مزامنة الأوامر: {str(e)}")
    
    async def detect_manually_closed_trades(self):
        """الكشف عن الصفقات التي تم إغلاقها يدوياً على المنصة"""
        try:
            # جلب تاريخ الأوامر الأخيرة
            recent_trades = await self.exchange.fetch_my_trades(symbol='BTC/USDT', since=int(time.time()) - 3600)
            
            for trade in recent_trades:
                if trade['order'] not in self.open_trades:
                    print(f"⚠️ تم اكتشاف صفقة تم إغلاقها يدوياً: {trade['order']}")
                    # إضافة الصفقة إلى السجل مع علامة الإغلاق اليدوي
                    self.add_manually_closed_trade(trade)
                    
        except Exception as e:
            print(f"خطأ في كشف الصفقات المغلقة يدوياً: {str(e)}")

class EnhancedAutoTradeManager(AutoTradeManager):
    def __init__(self, exchange, min_confidence=0.7, risk_per_trade=0.02, leverage=10):
        super().__init__(exchange, min_confidence, risk_per_trade, leverage)
        
        # إضافة المديرين الجدد
        self.trade_sync = TradeSyncManager(exchange)
        self.profit_calculator = ProfitCalculator(exchange)
        self.order_manager = EnhancedOrderManager(exchange)
        
        # مهمة دورية للمزامنة
        asyncio.create_task(self.periodic_sync())
    
    async def periodic_sync(self):
        """مزامنة دورية مع البورصة"""
        while True:
            try:
                for symbol in list(self.open_trades.keys()):
                    await self.trade_sync.sync_with_exchange(symbol)
                
                await asyncio.sleep(300)  # مزامنة كل 5 دقائق
              # التحقق من الصفقات التي قد تكون وصلت لوقف الخسارة
                for trade_id, trade in list(self.open_trades.items()):
                    current_price = await auto_trade_manager.get_current_price(trade['symbol'])
                    
                    if trade['side'] == 'BUY' and current_price <= trade['sl_level']:
                        # تم تفعيل وقف الخسارة لصفقة شراء
                        await self.handle_stop_loss(trade_id, current_price)
                    elif trade['side'] == 'SELL' and current_price >= trade['sl_level']:
                        # تم تفعيل وقف الخسارة لصفقة بيع
                        await self.handle_stop_loss(trade_id, current_price)
                
                await asyncio.sleep(60)  # التحقق كل دقيقة
                
            except Exception as e:
                print(f"❌ خطأ في المزامنة الدورية: {str(e)}")
                await asyncio.sleep(60)
    
    async def close_trade(self, trade_id, reason="manual"):
        """إغلاق صفقة وإزالتها من القائمة مع تسجيل سبب الإغلاق"""
        try:
            if trade_id in self.open_trades:
                trade = self.open_trades[trade_id]
                
                # تسجيل سبب الإغلاق
                trade['close_reason'] = reason
                trade['close_time'] = datetime.now()
                
                # حفظ بيانات الصفقة المغلقة للسجلات
                await self.save_closed_trade(trade_id, trade)
                
                # إزالة الصفقة من القائمة النشطة
                del self.open_trades[trade_id]
                
                # حفظ الحالة بعد الإغلاق
                self.save_state()
                
                print(f"✅ تم إغلاق الصفقة {trade_id} بسبب: {reason}")
                return True
            else:
                print(f"⚠️ الصفقة {trade_id} غير موجودة")
                return False
                
        except Exception as e:
            print(f"❌ خطأ في إغلاق الصفقة {trade_id}: {str(e)}")
            return False



class ProfitCalculator:
    def __init__(self, exchange):
        self.exchange = exchange
        
    async def calculate_pnl(self, symbol, side, entry_price, exit_price, size, leverage=1):
        """حسب الأرباح/الخسائر مع مراعاة الرافعة المالية"""
        try:
            # الحصول على معلومات الزوج
            market = self.exchange.market(symbol)
            contract_size = market['contractSize'] if 'contractSize' in market else 1
            
            # حساب قيمة النقطة
            ticker = await self.exchange.fetch_ticker(symbol)
            tick_size = market['precision']['price'] if 'precision' in market else 0.01
            
            # حساب الربح/الخسارة الأساسي
            if side == 'long':
                raw_pnl = (exit_price - entry_price) * size * contract_size
            else:
                raw_pnl = (entry_price - exit_price) * size * contract_size
            
            # تطبيق الرافعة المالية
            leveraged_pnl = raw_pnl * leverage
            
            # حساب الرسوم
            fee = await self.calculate_fee(symbol, size, entry_price, exit_price)
            
            # صافي الربح/الخسارة
            net_pnl = leveraged_pnl - fee
            
            return {
                'raw': raw_pnl,
                'leveraged': leveraged_pnl,
                'fee': fee,
                'net': net_pnl,
                'roe': (net_pnl / (size * entry_price * contract_size)) * 100 * leverage
            }
            
        except Exception as e:
            print(f"خطأ في حساب الأرباح/الخسائر: {str(e)}")
            return None
    
    async def calculate_fee(self, symbol, size, entry_price, exit_price):
        """حساب الرسوم الدقيقة"""
        # Binance تفرض رسوم 0.04% على العقود الآجلة :cite[4]
        fee_rate = 0.0004
        trade_value = (entry_price * size) + (exit_price * size)
        return trade_value * fee_rate




# ====== دوال الأوامر الجديدة ======
async def auto_trade_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """تفعيل/تعطيل التداول التلقائي"""
    if update.effective_user.id != int(AUTHORIZED_USER_ID):
        await update.message.reply_text("❌ ليس لديك صلاحية استخدام هذا الأمر.")
        return
    
    if context.args and context.args[0].lower() in ['on', 'تفعيل', 'تشغيل']:
        auto_trade_manager.auto_trading_enabled = True
        await update.message.reply_text("✅ تم تفعيل التداول التلقائي")
        await send_telegram_message_async("🟢 التداول التلقائي مفعل - سيفتح الصفقات تلقائياً عند الإشارات القوية")
    else:
        auto_trade_manager.auto_trading_enabled = False
        await update.message.reply_text("❌ تم تعطيل التداول التلقائي")
        await send_telegram_message_async("🔴 التداول التلقائي معطل")

async def set_risk_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """تعديل نسبة المخاطرة"""
    if update.effective_user.id != int(AUTHORIZED_USER_ID):
        await update.message.reply_text("❌ ليس لديك صلاحية استخدام هذا الأمر.")
        return
    
    if context.args and context.args[0].replace('%', '').isdigit():
        risk = float(context.args[0].replace('%', '')) / 100
        if 0.001 <= risk <= 0.1:  # بين 0.1% و 10%
            auto_trade_manager.risk_per_trade = risk
            await update.message.reply_text(f"✅ تم تعديل نسبة المخاطرة إلى {risk*100:.1f}%")
        else:
            await update.message.reply_text("❌ نسبة المخاطرة يجب أن تكون بين 0.1% و 10%")
    else:
        await update.message.reply_text(f"📊 نسبة المخاطرة الحالية: {auto_trade_manager.risk_per_trade*100:.1f}%")

async def state_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """إدارة حالة التداول (حفظ/تحميل)"""
    if update.effective_user.id != int(AUTHORIZED_USER_ID):
        await update.message.reply_text("❌ ليس لديك صلاحية استخدام هذا الأمر.")
        return
    
    if context.args and context.args[0].lower() == 'save':
        auto_trade_manager.save_state()
        await update.message.reply_text("✅ تم حفظ حالة التداول الحالية")
        
    elif context.args and context.args[0].lower() == 'load':
        try:
            auto_trade_manager.load_state()
            if auto_trade_manager.open_trades:
                await update.message.reply_text(f"✅ تم تحميل {len(auto_trade_manager.open_trades)} صفقة نشطة")
            else:
                await update.message.reply_text("✅ تم تحميل الحالة ولكن لا توجد صفقات نشطة")
        except Exception as e:
            await update.message.reply_text(f"❌ فشل تحميل الحالة: {str(e)}")
        
    elif context.args and context.args[0].lower() == 'verify':
        try:
            await auto_trade_manager.verify_active_trades()
            await update.message.reply_text("✅ تم التحقق من الصفقات النشطة مع البورصة")
        except Exception as e:
            await update.message.reply_text(f"❌ فشل التحقق من الصفقات: {str(e)}")
        
    else:
        await update.message.reply_text(
            "📋 أوامر إدارة الحالة:\n"
            "/state save - حفظ الحالة الحالية\n"
            "/state load - تحميل الحالة السابقة\n"
            "/state verify - التحقق من الصفقات مع البورصة"
        )

async def trades_status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """عرض حالة الصفقات المفتوحة بشكل مفصل مع الأرباح والخسائر"""
    if update.effective_user.id != int(AUTHORIZED_USER_ID):
        await update.message.reply_text("⛔️ **ليس لديك صلاحية استخدام هذا الأمر.**")
        return

    # التحقق من وجود صفقات محملة
    if not hasattr(auto_trade_manager, 'open_trades'):
        await update.message.reply_text("❌ **لم يتم تهيئة مدير التداول بعد.**")
        return

    # محاولة تحميل الحالة إذا كانت القائمة فارغة
    if not auto_trade_manager.open_trades:
        try:
            auto_trade_manager.load_state()
        except Exception as e:
            await update.message.reply_text(f"❌ **فشل تحميل الحالة:** {str(e)}")
            return
            
    if not auto_trade_manager.open_trades:
        await update.message.reply_text(
            "📭 **لا توجد صفقات مفتوحة حالياً**\n\n"
            "💡 **الأسباب المحتملة:**\n"
            "• لم يتم فتح أي صفقات بعد\n"
            "• تم إغلاق جميع الصفقات\n"
            "• مشكلة في تحميل الحالة السابقة\n"
            "• الصفقات غير متزامنة مع البورصة\n\n"
            "🔧 **الحلول المقترحة:**\n"
            "• استخدام `/state load` لإعادة تحميل الحالة\n"
            "• استخدام `/state verify` للتحقق من الصفقات مع البورصة\n"
            "• فتح صفقات جديدة"
        )
        return

   
    try:
        message = "📊 **الصفقات النشطة الحالية**\n\n"
        total_unrealized_pnl = 0
        total_invested = 0
        total_potential_profit = 0
        total_potential_loss = 0

        for trade_id, trade in auto_trade_manager.open_trades.items():
            current_price = await auto_trade_manager.get_current_price(trade['symbol'])
            invested_value = trade['entry_price'] * trade['size']
            total_invested += invested_value

            # حساب الأرباح/الخسائر غير المحققة
            if trade['side'] == "BUY":
                unrealized_pnl = (current_price - trade['entry_price']) * trade['size']
                sl_loss = (trade['sl_level'] - trade['entry_price']) * trade['size']
                sl_distance_pct = ((trade['entry_price'] - trade['sl_level']) / trade['entry_price']) * 100
                current_change = ((current_price - trade['entry_price']) / trade['entry_price']) * 100
            else:
                unrealized_pnl = (trade['entry_price'] - current_price) * trade['size']
                sl_loss = (trade['entry_price'] - trade['sl_level']) * trade['size']
                sl_distance_pct = ((trade['sl_level'] - trade['entry_price']) / trade['entry_price']) * 100
                current_change = ((trade['entry_price'] - current_price) / trade['entry_price']) * 100

            unrealized_pnl_percent = (unrealized_pnl / invested_value) * 100
            total_unrealized_pnl += unrealized_pnl

            # حساب الأرباح المحتملة لكل هدف
            target_profits = []
            for i, tp_level in enumerate(trade['tp_levels']):
                if trade['side'] == "BUY":
                    profit = (tp_level - trade['entry_price']) * trade['size']
                    profit_percent = ((tp_level - trade['entry_price']) / trade['entry_price']) * 100
                else:
                    profit = (trade['entry_price'] - tp_level) * trade['size']
                    profit_percent = ((trade['entry_price'] - tp_level) / trade['entry_price']) * 100
                target_profits.append((profit, profit_percent))
                total_potential_profit += profit

            total_potential_loss += abs(sl_loss)

            # تحديد عدد الأهداف المحققة
            achieved_targets = 0
            target_status = []
            for i, (tp_level, (profit, profit_percent)) in enumerate(zip(trade['tp_levels'], target_profits)):
                if (trade['side'] == "BUY" and current_price >= tp_level) or \
                   (trade['side'] == "SELL" and current_price <= tp_level):
                    achieved_targets += 1
                    target_status.append(
                        f"✅ الهدف {i+1}: `{tp_level:.6f}` (+{profit:.2f} USDT | +{profit_percent:.2f}%)"
                    )
                else:
                    if trade['side'] == "BUY":
                        remaining = ((tp_level - current_price) / current_price) * 100
                    else:
                        remaining = ((current_price - tp_level) / current_price) * 100
                    target_status.append(
                        f"⏳ الهدف {i+1}: `{tp_level:.6f}` (+{profit:.2f} USDT | +{profit_percent:.2f}%) "
                        f"- متبقي: {remaining:.2f}%"
                    )

            pnl_emoji = "🟢" if unrealized_pnl >= 0 else "🔴"
            direction_emoji = "🔺" if trade['side'] == "BUY" else "🔻"
            direction_arabic = "شراء" if trade['side'] == "BUY" else "بيع"

            # تنسيق الصفقة
            message += (
                f"{pnl_emoji} **{trade['symbol']}** {direction_emoji}\n"
                f"━━━━━━━━━━━━━━━\n"
                f"🆔 **المعرف:** `{trade_id}`\n"
                f"📈 **النوع:** {direction_arabic}\n"
                f"📊 **الحجم:** {trade['size']:.6f}\n"
                f"💰 **القيمة الاستثمارية:** {invested_value:.2f} USDT\n"
                f"🏷️ **سعر الدخول:** {trade['entry_price']:.6f}\n"
                f"💹 **السعر الحالي:** {current_price:.6f}\n"
                f"📉 **التغير الحالي:** {current_change:+.2f}%\n"
                f"💵 **الأرباح/الخسائر:** {unrealized_pnl:+.2f} USDT ({unrealized_pnl_percent:+.2f}%)\n"
                f"🛑 **وقف الخسارة:** {trade['sl_level']:.6f} "
                f"(-{abs(sl_distance_pct):.2f}% | خسارة محتملة: {sl_loss:+.2f} USDT)\n"
                f"🤝 **الثقة:** {trade['signal_confidence']:.1%}\n"
                f"🎯 **الأهداف المحققة:** {achieved_targets}/{len(trade['tp_levels'])}\n"
            )

            for status in target_status:
                message += f"  {status}\n"

            if trade['tp_levels']:
                total_tp_profit = sum(profit for profit, _ in target_profits)
                message += f"💎 **الربح الإجمالي المتوقع:** {total_tp_profit:.2f} USDT\n"

                if trade['side'] == "BUY":
                    next_target = min([tp for tp in trade['tp_levels'] if tp > current_price] or [max(trade['tp_levels'])])
                    target_distance = ((next_target - current_price) / current_price) * 100
                else:
                    next_target = max([tp for tp in trade['tp_levels'] if tp < current_price] or [min(trade['tp_levels'])])
                    target_distance = ((current_price - next_target) / current_price) * 100

                message += f"📏 **المسافة للهدف التالي:** {target_distance:.2f}%\n"

            message += f"🕒 **وقت الدخول:** {trade['timestamp'].strftime('%Y-%m-%d %H:%M')}\n"
            message += f"⏱️ **المدة:** {(datetime.now() - trade['timestamp']).total_seconds() / 3600:.1f} ساعة\n"
            message += "━━━━━━━━━━━━━━━━━━━━\n\n"

        total_profit_percent = (total_unrealized_pnl / total_invested * 100) if total_invested > 0 else 0
        risk_reward_ratio = (total_potential_profit / total_potential_loss) if total_potential_loss > 0 else 0

        message += (
            "📊 **الملخص العام**\n"
            "━━━━━━━━━━━━━━━\n"
            f"📂 **عدد الصفقات النشطة:** {len(auto_trade_manager.open_trades)}\n"
            f"💵 **إجمالي الاستثمار:** {total_invested:.2f} USDT\n"
            f"📈 **إجمالي الأرباح/الخسائر:** {total_unrealized_pnl:+.2f} USDT\n"
            f"📊 **نسبة العائد الإجمالية:** {total_profit_percent:+.2f}%\n"
            f"🎯 **إجمالي الأرباح المحتملة:** {total_potential_profit:.2f} USDT\n"
            f"🛑 **إجمالي الخسارة المحتملة:** {total_potential_loss:.2f} USDT\n"
            f"⚖️ **نسبة المكافأة/المخاطرة:** {risk_reward_ratio:.2f}:1\n"
            f"🕰️ **آخر تحديث:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

        # تقسيم الرسائل الطويلة
        if len(message) > 4096:
            message_parts = []
            current_part = ""

            for line in message.split('\n'):
                if len(current_part) + len(line) + 1 > 4000:
                    message_parts.append(current_part)
                    current_part = line + '\n'
                else:
                    current_part += line + '\n'
            if current_part:
                message_parts.append(current_part)

            for part in message_parts:
                await update.message.reply_text(part, parse_mode='Markdown')
                await asyncio.sleep(0.5)
        else:
            await update.message.reply_text(message, parse_mode='Markdown')

    except Exception as e:
        error_msg = f"❌ خطأ في عرض الصفقات النشطة: {str(e)}"
        logging.error(error_msg)
        await update.message.reply_text(error_msg)



# دالة مساعدة للحصول على حالة الأهداف
def get_target_status(current_price, tp_levels, side, entry_price):
    """الحصول على حالة تحقيق الأهداف"""
    status = []
    achieved = 0
    
    for i, tp in enumerate(tp_levels):
        if (side == "BUY" and current_price >= tp) or (side == "SELL" and current_price <= tp):
            profit_percent = ((tp - entry_price) / entry_price * 100) if side == "BUY" else ((entry_price - tp) / entry_price * 100)
            status.append(f"✅ الهدف {i+1} محقق: {tp:.6f} (+{profit_percent:.2f}%)")
            achieved += 1
        else:
            profit_percent = ((tp - entry_price) / entry_price * 100) if side == "BUY" else ((entry_price - tp) / entry_price * 100)
            if side == "BUY":
                remaining = ((tp - current_price) / current_price) * 100
            else:
                remaining = ((current_price - tp) / current_price) * 100
            status.append(f"⏳ الهدف {i+1}: {tp:.6f} (+{profit_percent:.2f}%) - متبقي: {remaining:.2f}%")
    
    return achieved, status



async def set_leverage_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """تعديل الرافعة المالية"""
    if update.effective_user.id != int(AUTHORIZED_USER_ID):
        await update.message.reply_text("❌ ليس لديك صلاحية استخدام هذا الأمر.")
        return
    
    if context.args and context.args[0].isdigit():
        leverage = int(context.args[0])
        if 1 <= leverage <= 100:
            auto_trade_manager.leverage = leverage
            await update.message.reply_text(f"✅ تم تعديل الرافعة المالية إلى {leverage}x")
            await send_telegram_message_async(f"📊 تم تعديل الرافعة المالية إلى {leverage}x")
        else:
            await update.message.reply_text("❌ الرافعة المالية يجب أن تكون بين 1 و 100")
    else:
        await update.message.reply_text(f"📊 الرافعة المالية الحالية: {auto_trade_manager.leverage}x")


async def signal_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != int(AUTHORIZED_USER_ID):
        await update.message.reply_text("❌ ليس لديك صلاحية استخدام هذا الأمر.")
        return

    if not context.args:
        await update.message.reply_text(
            "يرجى إدخال رمز أو عدة رموز (حتى 4). مثال:\n/signal BTCUSDT ETHUSDT"
        )
        return

    # إضافة معلومات عن وضع التداول الحالي
    mode_status = "فعلي 🟢" if TRADING_MODE == "real" else "تجريبي 🔵"
    await update.message.reply_text(f"🎮 وضع التداول الحالي: {mode_status}")

    symbols = [arg.upper() for arg in context.args[:4]]
    
    for symbol in symbols:
        try:
            # إرسال إشعار بدء التحليل مع معلومات الوضع
            await send_telegram_message_async(
                f"🔍 بدء تحليل {symbol}...\n"
                f"🎮 الوضع: {'فعلي 🟢' if TRADING_MODE == 'real' else 'تجريبي 🔵'}"
            )
            
            coin_symbol = symbol.split("USDT")[0]
            
            # 1. جلب وإرسال مؤشر الخوف والجشع الخاص بالعملة
            coin_sentiment = get_coin_sentiment(coin_symbol)
            
            if coin_sentiment:
                status_emoji = "😱" if coin_sentiment['value'] < 25 else \
                              "😨" if coin_sentiment['value'] < 45 else \
                              "😐" if coin_sentiment['value'] < 55 else \
                              "😊" if coin_sentiment['value'] < 75 else "🤑"
                
                # دمج المؤشر مع التحليل الفني
                market_context = ""
                if coin_sentiment['value'] < 30:
                    market_context = "⚠️ حالة خوف شديد - فرص شراء محتملة"
                elif coin_sentiment['value'] > 70:
                    market_context = "⚠️ حالة جشع شديد - احذر من التصحيح"
                
                await send_telegram_message_async(
                    f"📊 مؤشر مشاعر {coin_symbol}:\n"
                    f"{status_emoji} {coin_sentiment['value']}/100 ({coin_sentiment['classification']})\n"
                    f"• مؤشر المشاعر: {coin_sentiment['sentiment_score']}/100\n"
                    f"• مؤشر جالاكス: {coin_sentiment['galaxy_score']}/100\n"
                    f"⌚ التحديث: {coin_sentiment['timestamp'].strftime('%Y-%m-%d %H:%M')} UTC\n\n"
                    f"{market_context}"
                )
            else:
                # استخدام المؤشر العام كبديل
                general_sentiment = get_fear_greed_index()
                if general_sentiment:
                    status_emoji = "😱" if general_sentiment['value'] < 25 else \
                                  "😨" if general_sentiment['value'] < 45 else \
                                  "😐" if general_sentiment['value'] < 55 else \
                                  "😊" if general_sentiment['value'] < 75 else "🤑"
                    
                    await send_telegram_message_async(
                        f"📊 مؤشر المشاعر العام لسوق العملات:\n"
                        f"{status_emoji} {general_sentiment['value']}/100 ({general_sentiment['classification']})\n"
                        f"⌚ التحديث: {general_sentiment['timestamp'].strftime('%Y-%m-%d %H:%M')} UTC"
                    )
                else:
                    await send_telegram_message_async("⚠️ تعذر جلب مؤشر المشاعر")
            
            # 2. جلب وإرسال أخبار العملة مع تحليل المشاعر
            news = get_crypto_news(coin_symbol, num_articles=3)
            if news:
                news_msg = f"📰 آخر أخبار {coin_symbol}:\n\n"
                for i, article in enumerate(news, 1):
                    news_msg += (
                        f"{i}. {article['sentiment']} {article['title']}\n"
                        f"   - المصدر: {article['source']} | منذ {article['hours_ago']} ساعة\n"
                        f"   - [رابط الخبر]({article['url']})\n\n"
                    )
                await send_telegram_message_async(news_msg)
            else:
                await send_telegram_message_async(f"ℹ️ لا توجد أخبار حديثة عن {coin_symbol}")
            
            # استدعاء الدالة الرئيسية مع الاستقبال الثالث للبيانات
            result = get_signal(symbol)
            
            # التعامل مع النتائج بشكل صحيح
            if isinstance(result, tuple):
                if len(result) == 3:
                    message_parts, image_path, signal_data = result
                elif len(result) == 2:
                    message_parts, image_path = result
                    signal_data = None
                else:
                    # إذا كان عدد العناصر غير متوقع
                    message_parts = result[0] if len(result) > 0 else ["❌ خطأ في معالجة النتائج"]
                    image_path = result[1] if len(result) > 1 else None
                    signal_data = result[2] if len(result) > 2 else None
            else:
                # إذا كانت النتيجة ليست tuple
                message_parts = [result] if isinstance(result, str) else ["❌ نوع غير متوقع للنتيجة"]
                image_path = None
                signal_data = None
            
            # إرسال الرسائل النصية
            for part in message_parts:
                await send_telegram_message_async(part)
            
            # إرسال الصورة إذا كانت متوفرة
            if image_path and os.path.exists(image_path):
                await send_telegram_photo_async(image_path, f"تحليل {symbol}")
                
            # توليد وإرسال التقرير الذكي إذا كانت هناك بيانات إشارة
            if signal_data:
                report = generate_intelligent_prediction_report(signal_data, symbol)
                await send_telegram_message_async(report)
            
            # إذا كان التداول التلقائي مفعلاً وكانت هناك إشارة قوية
            if (auto_trade_manager.auto_trading_enabled and 
                signal_data and 
                signal_data.get('avg_confidence', 0) >= auto_trade_manager.min_confidence):
                
                # إضافة تحذير إذا كان الوضع تجريبي
                mode_warning = ""
                if TRADING_MODE == "demo":
                    mode_warning = "⚠️ <b>ملاحظة:</b> الوضع تجريبي - لن يتم فتح صفقة حقيقية\n"
                
                message = (
                    f"⚡ <b>إشارة قوية تم اكتشافها لـ {symbol}</b>\n"
                    f"{mode_warning}"
                    f"📊 الاتجاه: {'صعودي 🔺' if signal_data['overall_direction'] == 'صعود' else 'هبوطي 🔻'}\n"
                    f"🎯 الثقة: {signal_data['avg_confidence']:.1%}\n"
                    f"🤖 سيتم محاكاة فتح صفقة تلقائية (الوضع تجريبي)"
                )
                
                # إذا كان الوضع فعلي، تغيير الرسالة
                if TRADING_MODE == "real":
                    message = (
                        f"⚡ <b>إشارة قوية تم اكتشافها لـ {symbol}</b>\n"
                        f"📊 الاتجاه: {'صعودي 🔺' if signal_data['overall_direction'] == 'صعود' else 'هبوطي 🔻'}\n"
                        f"🎯 الثقة: {signal_data['avg_confidence']:.1%}\n"
                        f"🤖 سيتم فتح صفقة تلقائية حقيقية"
                    )
                
                await send_telegram_message_async(message)
                
                # محاولة فتح صفقة تلقائية
                trade_id = await auto_trade_manager.check_and_open_trade(symbol, signal_data)
                if trade_id:
                    mode_info = "محاكاة" if TRADING_MODE == "demo" else "حقيقية"
                    await send_telegram_message_async(f"✅ تم فتح صفقة تلقائية {mode_info} برقم: {trade_id}")
                else:
                    await send_telegram_message_async("⚠️ لم يتم فتح صفقة تلقائية بسبب عدم استيفاء الشروط")
                    
        except Exception as e:
            error_msg = f"❌ حدث خطأ أثناء تحليل {symbol}: {str(e)}"
            logging.error(error_msg)
            await send_telegram_message_async(error_msg)


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """عرض تعليمات استخدام البوت والأوامر المتاحة مع الشرح"""
    # تحميل الحالة السابقة عند البدء
    auto_trade_manager.load_state()
    
    # التحقق من الصفقات النشطة مع البورصة
    await auto_trade_manager.verify_active_trades()
    
    welcome_message = f"""
🤖 <b>بوت التداول الآلي المتقدم</b>

📋 <b>الأوامر المتاحة:</b>

🔹 <b>/signal [رمز العملة]</b>  
تحليل فني لعملة معينة باستخدام الذكاء الاصطناعي (مثال: <code>/signal BTCUSDT</code>)

🔹 <b>/autotrade on</b>  
تفعيل التداول التلقائي (سيبدأ البوت في فتح صفقات تلقائية)

🔹 <b>/autotrade off</b>  
تعطيل التداول التلقائي (إيقاف فتح صفقات جديدة تلقائياً)

🔹 <b>/risk [نسبة]</b>  
تعديل نسبة المخاطرة لكل صفقة (مثال: <code>/risk 2%</code> لتحديد 2% مخاطرة)

🔹 <b>/trades</b>  
عرض جميع الصفقات المفتوحة حالياً

🔹 <b>/status</b>  
عرض حالة التداول الحالية ومعلومات الحساب بسرعة

🔹 <b>/system</b>  
عرض حالة النظام الخاص بالبوت (الوقت – السيرفر – الاستجابة)

🔹 <b>/capital</b>  
عرض وتعديل رأس المال المخصص للتداول (أو نسبة استخدام رأس المال)

🔹 <b>/balance</b>  
عرض رصيد الحساب الحالي في باينانس (أو الحساب التجريبي)

🔹 <b>/help</b>  
عرض هذه التعليمات مرة أخرى

⚙️ <b>إعدادات التداول الحالية:</b>  
• <b>الوضع:</b> {'فعلي 🟢' if TRADING_MODE == 'real' else 'تجريبي 🔵'}  
• <b>نسبة المخاطرة:</b> {auto_trade_manager.risk_per_trade*100:.1f}%  
• <b>التداول التلقائي:</b> {'مفعل 🟢' if auto_trade_manager.auto_trading_enabled else 'معطل 🔴'}  

📢 <b>ملاحظة:</b>  
التداول يحمل مخاطر عالية. استخدم هذا البوت بحذر ولأغراض تعليمية فقط.
    """
    await update.message.reply_text(welcome_message, parse_mode='HTML')


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """عرض تعليمات استخدام البوت (نسخة مختصرة)"""
    await start_command(update, context)


async def get_balance_info():
    """الحصول على معلومات الرصيد من البورصة"""
    try:
        if TRADING_MODE == "real":
            try:
                # استدعاء fetch_balance بشكل صحيح باستخدام await
                balance = await exchange.fetch_balance()

                # الحصول على رصيد USDT
                usdt_balance = 0
                free_balance = 0
                used_balance = 0

                if 'USDT' in balance['total']:
                    usdt_balance = float(balance['total']['USDT'])
                    free_balance = float(balance['free'].get('USDT', 0))
                    used_balance = float(balance['used'].get('USDT', 0))

                return {
                    'total_balance': usdt_balance,
                    'free_balance': free_balance,
                    'locked_balance': used_balance,
                    'total_pnl': 0,
                    'pnl_percentage': 0,
                    'max_balance': usdt_balance,
                    'max_risk': usdt_balance * auto_trade_manager.risk_per_trade,
                    'current_risk': 0
                }
            except Exception as e:
                logging.error(f"Error fetching real balance: {str(e)}")
                return get_default_balance_info()
        else:
            return get_default_balance_info()
    except Exception as e:
        logging.error(f"Error in get_balance_info: {str(e)}")
        return get_default_balance_info()


def check_required_methods():
    """التحقق من وجود جميع الدوال المطلوبة"""
    required_methods = [
        'start_monitoring_system',
        'stop_all_tasks',
        'monitor_and_report_trades',
        'auto_update_trades_status',
        'send_hourly_detailed_report'
    ]
    
    missing_methods = []
    for method in required_methods:
        if not hasattr(auto_trade_manager, method):
            missing_methods.append(method)
    
    if missing_methods:
        print(f"❌ الدوال التالية مفقودة: {', '.join(missing_methods)}")
        return False
    
    return True

# في الدالة الرئيسية
if not check_required_methods():
    print("❌ لا يمكن بدء التشغيل بسبب دوال مفقودة")
    exit(1)

async def system_status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """عرض حالة النظام والإعدادات الحالية"""
    if update.effective_user.id != int(AUTHORIZED_USER_ID):
        await update.message.reply_text("❌ ليس لديك صلاحية استخدام هذا الأمر.")
        return
    
    # الحصول على معلومات النظام
    system_info = await get_system_info()
    
    status_message = f"""
🖥️ **حالة النظام والإعدادات**

🤖 **معلومات البوت:**
- حالة التشغيل: ✅ يعمل
- وقت التشغيل: {system_info['uptime']}
- إصدار النظام: {system_info['version']}
- الذاكرة المستخدمة: {system_info['memory_usage']}%
- مساحة التخزين: {system_info['storage_usage']}%

⚙️ **إعدادات التداول:**
- الوضع: {'فعلي 🟢' if TRADING_MODE == 'real' else 'تجريبي 🔵'}
- نسبة المخاطرة: {auto_trade_manager.risk_per_trade*100:.1f}%
- الحد الأدنى للثقة: {auto_trade_manager.min_confidence*100:.1f}%
- التداول التلقائي: {'مفعل 🟢' if auto_trade_manager.auto_trading_enabled else 'معطل 🔴'}

📊 **الأداء:**
- الصفقات النشطة: {len(auto_trade_manager.open_trades)}
- آخر تحديث: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    """
    
    await update.message.reply_text(status_message, parse_mode='Markdown')

async def capital_status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """عرض حالة رأس المال والرصيد"""
    if update.effective_user.id != int(AUTHORIZED_USER_ID):
        await update.message.reply_text("❌ ليس لديك صلاحية استخدام هذا الأمر.")
        return
    
    # الحصول على معلومات الرصيد
    balance_info = await get_balance_info()
    
    # تحديد مصدر البيانات
    data_source = "فعلي من البورصة 🟢" if TRADING_MODE == "real" else "تجريبي (قيم افتراضية) 🔵"
    
    capital_message = f"""
💰 **حالة رأس المال والرصيد ({data_source})**

💵 **الرصيد المتاح:**
- الرصيد الكلي: {balance_info['total_balance']:,.2f} USDT
- الرصيد المتاح: {balance_info['free_balance']:,.2f} USDT
- الرصيد المُقيد: {balance_info['locked_balance']:,.2f} USDT

📈 **أداء المحفظة:**
- إجمالي الربح/الخسارة: {balance_info['total_pnl']:+,.2f} USDT
- نسبة الربح/الخسارة: {balance_info['pnl_percentage']:+,.2f}%

🔒 **المخاطر:**
- الحد الأقصى للمخاطرة: {balance_info['max_risk']:,.2f} USDT
- المخاطرة الحالية: {balance_info['current_risk']:,.2f} USDT

🔄 **آخر تحديث:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    """
    
    await update.message.reply_text(capital_message, parse_mode='Markdown')

async def get_system_info():
    """الحصول على معلومات النظام"""
    try:
        # حساب وقت التشغيل
        startup_time = getattr(auto_trade_manager, 'startup_time', datetime.now())
        uptime = datetime.now() - startup_time
        uptime_str = str(uptime).split('.')[0]  # إزالة الكسور من الثواني
        
        # الحصول على استخدام الذاكرة (إذا كانت psutil متاحة)
        memory_usage = 0
        storage_usage = 0
        
        try:
            process = psutil.Process(os.getpid())
            memory_usage = process.memory_info().rss / 1024 / 1024  # تحويل إلى MB
            
            # الحصول على استخدام التخزين
            storage = psutil.disk_usage('/')
            storage_usage = (storage.used / storage.total) * 100
        except ImportError:
            # إذا لم تكن psutil مثبتة
            memory_usage = 0
            storage_usage = 0
        
        return {
            'uptime': uptime_str,
            'version': '2.0.0',
            'memory_usage': round(memory_usage, 2),
            'storage_usage': round(storage_usage, 2)
        }
    except Exception as e:
        logging.error(f"Error getting system info: {str(e)}")
        # في حالة الخطأ، نعود بقيم افتراضية
        return {
            'uptime': 'غير معروف',
            'version': '2.0.0',
            'memory_usage': 0,
            'storage_usage': 0
        }


def get_default_balance_info():
    """إرجاع معلومات رصيد افتراضية"""
    return {
        'total_balance': 10000,
        'free_balance': 8000,
        'locked_balance': 2000,
        'total_pnl': 350.50,
        'pnl_percentage': 3.5,
        'max_balance': 10500,
        'max_risk': 10000 * auto_trade_manager.risk_per_trade,
        'current_risk': 150
    }

async def send_startup_message():
    """إرسال رسالة بدء التشغيل"""
    try:
        system_info = await get_system_info()
        balance_info = await get_balance_info()
        
        # تحديد مصدر بيانات الرصيد
        balance_source = "فعلي من البورصة 🟢" if TRADING_MODE == "real" else "تجريبي (قيم افتراضية) 🔵"
        
        startup_message = f"""
🚀 **تم تشغيل بوت التداول الآلي بنجاح**

🖥️ **معلومات النظام:**
- وقت التشغيل: {system_info['uptime']}
- الذاكرة المستخدمة: {system_info['memory_usage']}%
- مساحة التخزين: {system_info['storage_usage']}%

💰 **حالة الرصيد ({balance_source}):**
- الرصيد الكلي: {balance_info['total_balance']:,.2f} USDT
- الرصيد المتاح: {balance_info['free_balance']:,.2f} USDT
- الرصيد المُقيد: {balance_info['locked_balance']:,.2f} USDT

📈 **أداء المحفظة:**
- إجمالي الربح/الخسارة: {balance_info['total_pnl']:+,.2f} USDT
- نسبة الربح/الخسارة: {balance_info['pnl_percentage']:+,.2f}%

⚙️ **الإعدادات الحالية:**
- وضع التداول: {'فعلي 🟢' if TRADING_MODE == 'real' else 'تجريبي 🔵'}
- نسبة المخاطرة: {auto_trade_manager.risk_per_trade*100:.1f}%
- الحد الأدنى للثقة: {auto_trade_manager.min_confidence*100:.1f}%
- التداول التلقائي: {'مفعل 🟢' if auto_trade_manager.auto_trading_enabled else 'معطل 🔴'}

📊 **المخاطر:**
- الحد الأقصى للمخاطرة: {balance_info['max_risk']:,.2f} USDT
- المخاطرة الحالية: {balance_info['current_risk']:,.2f} USDT

📋 **الأوامر المتاحة:**
- /start - عرض التعليمات
- /status - حالة النظام
- /capital - حالة رأس المال
- /signal - تحليل عملة
- /autotrade - التحكم بالتداول التلقائي
- /risk - تعديل المخاطرة
- /trades - الصفقات النشطة

⏰ **وقت التشغيل:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        await send_telegram_message_async(startup_message)
    except Exception as e:
        logging.error(f"Error sending startup message: {str(e)}")
        # إرسال رسالة بديلة في حالة الخطأ
        simple_message = f"""
🤖 تم تشغيل بوت التداول الآلي بنجاح!
📊 وضع التداول: {'فعلي 🟢' if TRADING_MODE == 'real' else 'تجريبي 🔵'}
💰 الرصيد: جاري التحقق...
🔄 آخر تحديث: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        await send_telegram_message_async(simple_message)


async def auto_update_trades_status(self):
    """تحديث حالة الصفقات تلقائياً مع التحقق من حالة الإغلاق"""
    try:
        while not self._is_shutting_down:
            try:
                if not self.open_trades:
                    # الانتظار مع التحقق من الإغلاق
                    for _ in range(300):
                        if self._is_shutting_down:
                            break
                        await asyncio.sleep(1)
                    continue
                
                # نسخ القائمة لتجنب تغييرها أثناء التكرار
                trades_to_check = list(self.open_trades.items())
                
                for trade_id, trade in trades_to_check:
                    if self._is_shutting_down:
                        break
                        
                    try:
                        current_price = await self.get_current_price(trade['symbol'])
                        
                        # التحقق إذا تم إغلاق الصفقة (وصل إلى STOP LOSS)
                        if ((trade['side'] == "BUY" and current_price <= trade['sl_level']) or
                            (trade['side'] == "SELL" and current_price >= trade['sl_level'])):
                            
                            # حساب PNL النهائي
                            if trade['side'] == "BUY":
                                pnl = (current_price - trade['entry_price']) * trade['size']
                            else:
                                pnl = (trade['entry_price'] - current_price) * trade['size']
                            
                            # إرسال إشعار الإغلاق
                            await send_telegram_message_async(
                                f"⚠️ **تم إغلاق الصفقة تلقائياً**\n"
                                f"• الزوج: {trade['symbol']}\n"
                                f"• المعرف: {trade_id}\n"
                                f"• السبب: وصل إلى وقف الخسارة\n"
                                f"• الأرباح/الخسائر النهائية: {pnl:+.2f} USDT\n"
                                f"• السعر النهائي: {current_price:.6f}"
                            )
                            
                            # إزالة الصفقة من القائمة
                            if trade_id in self.open_trades:
                                del self.open_trades[trade_id]
                        
                        # التحقق من تحقيق جميع الأهداف
                        else:
                            achieved_targets = 0
                            for tp_level in trade['tp_levels']:
                                if ((trade['side'] == "BUY" and current_price >= tp_level) or
                                    (trade['side'] == "SELL" and current_price <= tp_level)):
                                    achieved_targets += 1
                            
                            # إذا تم تحقيق جميع الأهداف
                            if achieved_targets == len(trade['tp_levels']):
                                if trade['side'] == "BUY":
                                    pnl = (current_price - trade['entry_price']) * trade['size']
                                else:
                                    pnl = (trade['entry_price'] - current_price) * trade['size']
                                
                                await send_telegram_message_async(
                                    f"🎯 **تم تحقيق جميع الأهداف**\n"
                                    f"• الزوج: {trade['symbol']}\n"
                                    f"• المعرف: {trade_id}\n"
                                    f"• الأرباح النهائية: {pnl:+.2f} USDT\n"
                                    f"• السعر النهائي: {current_price:.6f}"
                                )
                                
                                if trade_id in self.open_trades:
                                    del self.open_trades[trade_id]
                    
                    except Exception as e:
                        print(f"❌ خطأ في التحقق من الصفقة {trade_id}: {str(e)}")
                        continue
                
                # الانتظار 5 دقائق قبل التحديث التالي مع التحقق من الإغلاق
                for _ in range(300):
                    if self._is_shutting_down:
                        break
                    await asyncio.sleep(1)
                    
            except asyncio.CancelledError:
                print("⏹️ تم إيقاف مهمة التحديث التلقائي")
                break
            except Exception as e:
                print(f"❌ خطأ في التحديث التلقائي: {str(e)}")
                # الانتظار دقيقة قبل إعادة المحاولة مع التحقق من الإغلاق
                for _ in range(60):
                    if self._is_shutting_down:
                        break
                    await asyncio.sleep(1)
                    
    except asyncio.CancelledError:
        print("⏹️ تم إلغاء مهمة التحديث التلقائي")
    except Exception as e:
        print(f"❌ خطأ غير متوقع في التحديث التلقائي: {str(e)}")

async def sync_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """مزامنة الصفقات مع البورصة"""
    if update.effective_user.id != int(AUTHORIZED_USER_ID):
        await update.message.reply_text("❌ ليس لديك صلاحية استخدام هذا الأمر.")
        return

    try:
        await auto_trade_manager.sync_with_binance()
        await update.message.reply_text("✅ تمت المزامنة مع البورصة بنجاح")
        
        # تحديث قائمة الصفقات بعد المزامنة
        if not auto_trade_manager.open_trades:
            await update.message.reply_text("📭 لا توجد صفقات نشطة بعد المزامنة")
        else:
            await update.message.reply_text(f"📊 يوجد {len(auto_trade_manager.open_trades)} صفقة نشطة")
            
    except Exception as e:
        await update.message.reply_text(f"❌ خطأ في المزامنة: {str(e)}")



import atexit
import signal

# دالة للتعامل مع الإغلاق النظيف
async def graceful_shutdown():
    """إيقاف جميع المهام بشكل صحيح قبل إغلاق البرنامج"""
    print("⏹️ جاري إغلاق البرنامج بشكل نظيف...")
    
    # إيقاف جميع المهام في مدير التداول
    if 'auto_trade_manager' in globals():
        await auto_trade_manager.stop_all_tasks()
    
    # إغلاق اتصال البورصة إذا كان موجوداً
    if 'exchange' in globals() and hasattr(exchange, 'close'):
        await exchange.close()
    
    print("✅ تم إغلاق البرنامج بشكل نظيف")

# تسجيل دالة الإغلاق لتعمل عند إنهاء البرنامج
def handle_exit():
    """معالجة إشارات الخروج"""
    asyncio.run(graceful_shutdown())

# التسجيل لمعالجة إشارات النظام
signal.signal(signal.SIGINT, lambda s, f: handle_exit())
signal.signal(signal.SIGTERM, lambda s, f: handle_exit())

# التسجيل للعمل عند انتهاء البرنامج
atexit.register(handle_exit)

# تعديل دالة إعداد ال handlers لإضافة الأوامر الجديدة
def setup_handlers(application):
    """إعداد جميع handlers للتطبيق"""
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("status", system_status_command))
    application.add_handler(CommandHandler("system", system_status_command))
    application.add_handler(CommandHandler("capital", capital_status_command))
    application.add_handler(CommandHandler("balance", capital_status_command))
    application.add_handler(CommandHandler("signal", signal_command))
    application.add_handler(CommandHandler("autotrade", auto_trade_command))
    application.add_handler(CommandHandler("risk", set_risk_command))
    application.add_handler(CommandHandler("trades", trades_status_command))
    application.add_handler(CommandHandler("leverage", set_leverage_command))
    application.add_handler(CommandHandler("state", state_command))



# ====== إصلاح الجزء الرئيسي للتشغيل ======
if __name__ == "__main__":
    # تسجيل وقت بدء التشغيل
    auto_trade_manager.startup_time = datetime.now()
    
    # إنشاء تطبيق Telegram
    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    
    # إعداد ال handlers
    setup_handlers(application)
    
    # بدء البوت
    print("🤖 بدء تشغيل بوت التداول الآلي...")
    
    # التحقق من وجود الدوال المطلوبة
    required_methods = [
        'start_monitoring_system',
        'stop_all_tasks',
        'monitor_and_report_trades',
        'auto_update_trades_status',
        'send_hourly_detailed_report'
    ]
    
    missing_methods = []
    for method in required_methods:
        if not hasattr(auto_trade_manager, method):
            missing_methods.append(method)
    
    if missing_methods:
        print(f"❌ الدوال التالية مفقودة: {', '.join(missing_methods)}")
        print("❌ لا يمكن بدء التشغيل بسبب دوال مفقودة")
        exit(1)
    
    # إنشاء حلقة أحداث جديدة
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    async def main_async():
        """الدالة الرئيسية غير المتزامنة"""
        try:
            # إرسال رسالة البدء
            await send_startup_message()
            
            # بدء نظام المراقبة
            await auto_trade_manager.start_monitoring_system()
            
            # بدء المزامنة الدورية لمدير التداول
            if hasattr(auto_trade_manager, 'start_periodic_sync'):
                await auto_trade_manager.start_periodic_sync()
            
            # بدء استقبال الأوامر من Telegram
            print("📡 بدء استقبال الأوامر من Telegram...")
            
            # استخدام run_polling بشكل منفصل
            await application.initialize()
            await application.start()
            await application.updater.start_polling()
            
            # البقاء في الحلقة حتى يتم إيقاف البوت
            while True:
                await asyncio.sleep(1)
                
        except Exception as e:
            print(f"❌ خطأ في الدالة الرئيسية: {str(e)}")
            raise e
    
    try:
        # تشغيل الدالة الرئيسية
        loop.run_until_complete(main_async())
        
    except KeyboardInterrupt:
        print("⏹️ إيقاف البوت بواسطة المستخدم...")
    except Exception as e:
        print(f"❌ خطأ في التشغيل: {str(e)}")
    finally:
        # تنظيف المهمة عند الخروج
        try:
            # إيقاف المزامنة الدورية لمدير التداول
            if hasattr(auto_trade_manager, 'stop_periodic_sync'):
                loop.run_until_complete(auto_trade_manager.stop_periodic_sync())
            
            # إيقاف جميع المهام
            if hasattr(auto_trade_manager, 'stop_all_tasks'):
                loop.run_until_complete(auto_trade_manager.stop_all_tasks())
            
            # إيقاف البوت
            if application.running:
                loop.run_until_complete(application.stop())
                loop.run_until_complete(application.shutdown())
        except Exception as e:
            print(f"❌ خطأ أثناء التنظيف: {str(e)}")
        finally:
            loop.close()
            print("✅ تم إغلاق حلقة الأحداث والخروج من البرنامج")