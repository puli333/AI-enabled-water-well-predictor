from flask import Flask, request, jsonify, session, redirect, url_for, send_from_directory
import importlib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import logging
import sqlite3
from datetime import datetime
from typing import Optional
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import shutil

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
CSV_PATH = os.path.join(DATA_DIR, 'cgwb_tables.csv')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import Flask-CORS
try:
    from flask_cors import CORS
    _CORS_AVAILABLE = True
except ImportError:
    CORS = None
    _CORS_AVAILABLE = False
    logger.warning(
        "Optional dependency 'flask-cors' is not installed. "
        "Install it via 'pip install flask-cors' to enable full CORS support."
    )

app = Flask(__name__, static_folder='.', static_url_path='')
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production-2024')

# Enable CORS for all routes with credentials if available
if _CORS_AVAILABLE and CORS:
    # Allow credentials and all origins for development environment
    CORS(app, supports_credentials=True, resources={r"/*": {"origins": "*"}})
else:
    # Basic CORS fallback if flask-cors is not installed
    @app.after_request
    def _apply_basic_cors(response):
        response.headers.setdefault('Access-Control-Allow-Origin', '*')
        response.headers.setdefault('Access-Control-Allow-Credentials', 'true')
        response.headers.setdefault(
            'Access-Control-Allow-Headers',
            'Content-Type,Authorization'
        )
        response.headers.setdefault(
            'Access-Control-Allow-Methods',
            'GET,POST,PUT,DELETE,OPTIONS'
        )
        return response

# Initialize models and encoders
models = {}
encoders = {}

# Database setup
DB_PATH = os.path.join(os.path.dirname(__file__), 'users.db')
PROFILE_PICS_DIR = os.path.join(os.path.dirname(__file__), 'static', 'profile_pics')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

# Ensure profile pictures directory exists
os.makedirs(PROFILE_PICS_DIR, exist_ok=True)

def init_db():
    """Initialize the SQLite database and create users table"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT,
            locality TEXT,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP
        )
    ''')
    
    # Add locality column if it doesn't exist (for existing databases)
    try:
        cursor.execute('ALTER TABLE users ADD COLUMN locality TEXT')
    except sqlite3.OperationalError:
        # Column already exists, ignore
        pass
    
    # Add profile_picture column if it doesn't exist
    try:
        cursor.execute('ALTER TABLE users ADD COLUMN profile_picture TEXT')
    except sqlite3.OperationalError:
        # Column already exists, ignore
        pass
    
    # Check if default users exist, if not create them
    cursor.execute('SELECT COUNT(*) FROM users WHERE username IN (?, ?)', ('admin', 'user'))
    count = cursor.fetchone()[0]
    
    if count == 0:
        # Create default users
        admin_hash = generate_password_hash('admin123')
        user_hash = generate_password_hash('user123')
        cursor.execute('''
            INSERT INTO users (username, email, locality, password_hash)
            VALUES (?, ?, ?, ?)
        ''', ('admin', 'admin@example.com', 'Default Locality', admin_hash))
        cursor.execute('''
            INSERT INTO users (username, email, locality, password_hash)
            VALUES (?, ?, ?, ?)
        ''', ('user', 'user@example.com', 'Default Locality', user_hash))
        logger.info("Created default users (admin/admin123 and user/user123)")
    
    conn.commit()
    conn.close()
    logger.info("Database initialized successfully")

def get_db_connection():
    """Get a database connection"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def get_user(username):
    """Get user from database"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE username = ?', (username.lower(),))
    user = cursor.fetchone()
    conn.close()
    return user

def update_user_profile(username, email=None, locality=None, profile_picture=None):
    """Update user profile information"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    updates = []
    values = []
    
    if email is not None:
        updates.append('email = ?')
        values.append(email.lower() if email else None)
    if locality is not None:
        updates.append('locality = ?')
        values.append(locality.strip() if locality else None)
    if profile_picture is not None:
        updates.append('profile_picture = ?')
        values.append(profile_picture)
    
    if updates:
        values.append(username.lower())
        cursor.execute(f'''
            UPDATE users 
            SET {', '.join(updates)}
            WHERE username = ?
        ''', values)
        conn.commit()
    
    conn.close()

def delete_user_account(username):
    """Delete user account and associated files"""
    user = get_user(username)
    if not user:
        return False
    
    # Delete profile picture if exists
    if user.get('profile_picture'):
        pic_path = os.path.join(PROFILE_PICS_DIR, user['profile_picture'])
        if os.path.exists(pic_path):
            try:
                os.remove(pic_path)
            except Exception as e:
                logger.error(f"Error deleting profile picture: {e}")
    
    # Delete user from database
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM users WHERE username = ?', (username.lower(),))
    conn.commit()
    conn.close()
    
    return True

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_user(username, email, locality, password_hash):
    """Create a new user in the database"""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT INTO users (username, email, locality, password_hash)
            VALUES (?, ?, ?, ?)
        ''', (username.lower(), email.lower() if email else None, locality.strip() if locality else None, password_hash))
        conn.commit()
        user_id = cursor.lastrowid
        conn.close()
        return user_id
    except sqlite3.IntegrityError:
        conn.close()
        raise ValueError("Username already exists")

def update_last_login(username):
    """Update last login timestamp"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE username = ?
    ''', (username.lower(),))
    conn.commit()
    conn.close()

# Initialize database on startup
# init_db() # Moved to main to ensure it runs once

def login_required(f):
    """Decorator to require login for routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session or not session['logged_in']:
            return jsonify({'error': 'Authentication required'}), 401
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    """Decorator to require admin login for routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session or not session['logged_in']:
            return jsonify({'error': 'Authentication required'}), 401
        if session.get('username', '').lower() != 'admin':
            return jsonify({'error': 'Admin access required'}), 403
        return f(*args, **kwargs)
    return decorated_function

def get_all_users():
    """Get all users from database (admin only)"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, username, email, locality, created_at, last_login
        FROM users
        ORDER BY created_at DESC
    ''')
    users = cursor.fetchall()
    conn.close()
    return users

def load_cgwb_data() -> Optional[pd.DataFrame]:
    """Attempt to load parsed CGWB CSV and coerce to a modeling dataframe.
    The PDF tables are heterogeneous; this function extracts any water level-like columns if present.
    Returns a dataframe with expected columns or None if not enough data.
    """
    try:
        if not os.path.isfile(CSV_PATH):
            return None
        df = pd.read_csv(CSV_PATH)
        if df.empty:
            return None
        # Create modeled features with best-effort mapping; many columns may not exist.
        # We synthesize geospatial and environmental features due to PDF limitations.
        n = len(df)
        rng = np.random.default_rng(42)
        modeled = pd.DataFrame({
            'latitude': rng.uniform(6.0, 37.0, n),
            'longitude': rng.uniform(68.0, 97.0, n),
            'soil_type': rng.choice(['sandy', 'clay', 'loam', 'silt'], n),
            'lithology': rng.choice(['granite', 'basalt', 'limestone', 'shale'], n),
            'land_use': rng.choice(['agriculture', 'forest', 'urban', 'grassland'], n),
            'rainfall_mm': rng.uniform(200, 2000, n),
            'slope_deg': rng.uniform(0, 30, n),
            'elevation_m': rng.uniform(0, 3000, n),
            'water_table_m': rng.uniform(1, 50, n),
            'distance_to_river_km': rng.uniform(0.1, 20, n),
            'ndvi': rng.uniform(0, 1, n)
        })
        # If any likely depth/water level columns exist, use them to adjust targets
        level_cols = [c for c in df.columns if 'water level' in c.lower() or 'wl' in c.lower() or 'depth' in c.lower()]
        if level_cols:
            levels = pd.to_numeric(df[level_cols[0]], errors='coerce')
            modeled['water_table_m'] = np.where(levels.notna(), np.clip(levels.values, 1, 60), modeled['water_table_m'])
        return modeled
    except Exception as ex:
        logger.warning(f"Failed to load/shape CGWB data: {ex}")
        return None


def create_sample_data():
    """Create sample training data for demonstration"""
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic data
    data = {
        'latitude': np.random.uniform(6.0, 37.0, n_samples),  # India latitude range
        'longitude': np.random.uniform(68.0, 97.0, n_samples),  # India longitude range
        'soil_type': np.random.choice(['sandy', 'clay', 'loam', 'silt'], n_samples),
        'lithology': np.random.choice(['granite', 'basalt', 'limestone', 'shale'], n_samples),
        'land_use': np.random.choice(['agriculture', 'forest', 'urban', 'grassland'], n_samples),
        'rainfall_mm': np.random.uniform(200, 2000, n_samples),
        'slope_deg': np.random.uniform(0, 30, n_samples),
        'elevation_m': np.random.uniform(0, 3000, n_samples),
        'water_table_m': np.random.uniform(1, 50, n_samples),
        'distance_to_river_km': np.random.uniform(0.1, 20, n_samples),
        'ndvi': np.random.uniform(0, 1, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Create target variables based on realistic relationships
    # Suitability: combination of factors
    suitability_score = (
        (df['rainfall_mm'] > 800) * 0.3 +
        (df['soil_type'].isin(['loam', 'sandy'])) * 0.2 +
        (df['lithology'].isin(['granite', 'basalt'])) * 0.2 +
        (df['water_table_m'] < 20) * 0.2 +
        (df['distance_to_river_km'] < 5) * 0.1
    )
    
    df['suitable'] = (suitability_score > 0.5).astype(int)
    
    # Depth prediction (deeper wells in dry areas, shallow in wet areas)
    df['depth_m'] = np.random.uniform(10, 100, n_samples)
    df.loc[df['rainfall_mm'] < 500, 'depth_m'] += 20
    df.loc[df['water_table_m'] > 30, 'depth_m'] += 15
    
    # Discharge prediction (higher in suitable areas)
    df['discharge_lps'] = np.random.uniform(0.5, 10, n_samples)
    df.loc[df['suitable'] == 1, 'discharge_lps'] *= 1.5
    
    # Quality index (0-100)
    df['quality_index'] = np.random.uniform(60, 95, n_samples)
    df.loc[df['suitable'] == 0, 'quality_index'] -= 20
    
    return df

def train_models():
    """Train machine learning models"""
    logger.info("Creating and training models...")
    # Prefer CGWB-parsed data if available; otherwise synthetic
    df = load_cgwb_data()
    if df is None or df.empty:
        logger.info("Using synthetic dataset (CGWB parsed data unavailable or empty)")
        df = create_sample_data()
    
    # Prepare features
    categorical_features = ['soil_type', 'lithology', 'land_use']
    numerical_features = ['latitude', 'longitude', 'rainfall_mm', 'slope_deg', 
                         'elevation_m', 'water_table_m', 'distance_to_river_km', 'ndvi']
    
    # Encode categorical variables
    for feature in categorical_features:
        le = LabelEncoder()
        # Handle case where the column doesn't exist in the data (e.g., if using only specific CGWB data columns)
        if feature in df.columns:
            df[f'{feature}_encoded'] = le.fit_transform(df[feature])
            encoders[feature] = le
        else:
            logger.warning(f"Feature '{feature}' not found in dataset. Skipping encoding.")
    
    # Filter for existing feature columns
    feature_columns = [f'{f}_encoded' for f in categorical_features if f in encoders] + numerical_features
    X = df[feature_columns]
    
    # Train suitability classifier
    y_suitable = df['suitable']
    models['suitability'] = RandomForestClassifier(n_estimators=100, random_state=42)
    models['suitability'].fit(X, y_suitable)
    
    # Train depth regressor
    y_depth = df['depth_m']
    models['depth'] = RandomForestRegressor(n_estimators=100, random_state=42)
    models['depth'].fit(X, y_depth)
    
    # Train discharge regressor
    y_discharge = df['discharge_lps']
    models['discharge'] = RandomForestRegressor(n_estimators=100, random_state=42)
    models['discharge'].fit(X, y_discharge)
    
    # Train quality regressor
    y_quality = df['quality_index']
    models['quality'] = RandomForestRegressor(n_estimators=100, random_state=42)
    models['quality'].fit(X, y_quality)
    
    logger.info("Models trained successfully!")

def preprocess_input(data):
    """Preprocess input data for prediction"""
    # Create a copy of the input data
    processed = data.copy()
    
    # Encode categorical variables
    categorical_features = ['soil_type', 'lithology', 'land_use']
    for feature in categorical_features:
        if feature in processed and feature in encoders:
            try:
                # Use encoder to transform the single value
                processed[f'{feature}_encoded'] = encoders[feature].transform([processed[feature]])[0]
            except ValueError:
                # Handle unseen categories by assigning a default (0)
                logger.warning(f"Unseen category '{processed[feature]}' for feature '{feature}'. Defaulting to 0.")
                processed[f'{feature}_encoded'] = 0
    
    # Prepare feature vector in the same order as training
    numerical_features = ['latitude', 'longitude', 'rainfall_mm', 'slope_deg', 
                         'elevation_m', 'water_table_m', 'distance_to_river_km', 'ndvi']
    
    # Use only encoded features for which encoders exist
    feature_columns = [f'{f}_encoded' for f in categorical_features if f in encoders] + numerical_features
    
    feature_vector = []
    for col in feature_columns:
        # Extract original feature name from encoded name
        original_feature = col.replace('_encoded', '')
        
        # Prioritize encoded value if available
        if col.endswith('_encoded') and col in processed:
            feature_vector.append(processed[col])
        elif original_feature in processed:
             # For numerical features, use the direct value
            feature_vector.append(processed[original_feature])
        else:
            # Default value for missing features
            feature_vector.append(0) 
            logger.warning(f"Missing input data for required feature '{original_feature}'. Defaulting to 0.")
    
    return np.array(feature_vector).reshape(1, -1)


def get_readymade_result(data: dict) -> dict:
    """Return predefined, India-centric results when ML results aren't available.
    Values are indicative and tailored per land_use type.
    """
    land_use = str(data.get('land_use', '')).strip().lower()
    presets = {
        'agriculture': {
            'suitable_probability': 0.78,
            'predicted_depth_m': 40.0,
            'predicted_discharge_lps': 6.0,
            'predicted_quality_index': 82.0,
        },
        'forest': {
            'suitable_probability': 0.72,
            'predicted_depth_m': 35.0,
            'predicted_discharge_lps': 5.5,
            'predicted_quality_index': 85.0,
        },
        'urban': {
            'suitable_probability': 0.42,
            'predicted_depth_m': 65.0,
            'predicted_discharge_lps': 3.0,
            'predicted_quality_index': 70.0,
        },
        'grassland': {
            'suitable_probability': 0.55,
            'predicted_depth_m': 50.0,
            'predicted_discharge_lps': 4.0,
            'predicted_quality_index': 75.0,
        }
    }

    preset = presets.get(land_use, {
        'suitable_probability': 0.5,
        'predicted_depth_m': 50.0,
        'predicted_discharge_lps': 4.0,
        'predicted_quality_index': 75.0,
    })

    suitable_flag = 1 if preset['suitable_probability'] >= 0.5 else 0
    return {
        'suitable': suitable_flag,
        'suitable_probability': float(preset['suitable_probability']),
        'predicted_depth_m': float(preset['predicted_depth_m']),
        'predicted_discharge_lps': float(preset['predicted_discharge_lps']),
        'predicted_quality_index': float(preset['predicted_quality_index']),
        'input_data': data,
        'source': 'readymade'
    }

@app.route('/signup', methods=['POST'])
def signup():
    """Signup endpoint"""
    try:
        data = request.get_json()
        if not data:
            logger.warning("Signup attempt with no data")
            return jsonify({'error': 'Invalid request. Please provide username, email, and password.'}), 400
            
        username = data.get('username', '').strip()
        email = data.get('email', '').strip()
        locality = data.get('locality', '').strip()
        password = data.get('password', '').strip()
        
        if not username or not password:
            return jsonify({'error': 'Username and password are required'}), 400
        
        if len(username) < 3:
            return jsonify({'error': 'Username must be at least 3 characters long'}), 400
        
        if len(password) < 6:
            return jsonify({'error': 'Password must be at least 6 characters long'}), 400
        
        # Check if user already exists
        existing_user = get_user(username)
        if existing_user:
            return jsonify({'error': 'Username already exists. Please choose a different username.'}), 400
        
        # Create new user
        password_hash = generate_password_hash(password)
        try:
            user_id = create_user(username, email, locality, password_hash)
            logger.info(f"New user created: {username} (ID: {user_id})")
            return jsonify({
                'success': True,
                'message': 'Account created successfully! You can now login.',
                'username': username
            }), 201
        except ValueError as e:
            return jsonify({'error': str(e)}), 400
            
    except Exception as e:
        logger.error(f"Signup error: {str(e)}", exc_info=True)
        return jsonify({'error': f'Signup failed: {str(e)}'}), 500

@app.route('/login', methods=['POST'])
def login():
    """Login endpoint"""
    try:
        data = request.get_json()
        if not data:
            logger.warning("Login attempt with no data")
            return jsonify({'error': 'Invalid request. Please provide username and password.'}), 400
            
        username = data.get('username', '').strip()
        password = data.get('password', '').strip()
        
        if not username or not password:
            logger.warning(f"Login attempt with missing credentials for user: {username}")
            return jsonify({'error': 'Username and password are required'}), 400
        
        # Lookup user from database
        user = get_user(username)
        if not user:
            logger.warning(f"Login attempt with unknown username: '{username}'")
            return jsonify({'error': 'Invalid username or password'}), 401
        
        # Check password
        password_match = check_password_hash(user['password_hash'], password)
        logger.info(f"Password check for {username}: {password_match}")
        
        if password_match:
            session['logged_in'] = True
            session['username'] = user['username']
            # Update last login timestamp
            update_last_login(username)
            logger.info(f"User {username} logged in successfully")
            return jsonify({
                'success': True,
                'message': 'Login successful',
                'username': user['username']
            })
        else:
            logger.warning(f"Login attempt with incorrect password for user: {username}")
            return jsonify({'error': 'Invalid username or password'}), 401
            
    except Exception as e:
        logger.error(f"Login error: {str(e)}", exc_info=True)
        return jsonify({'error': f'Login failed: {str(e)}'}), 500

@app.route('/logout', methods=['POST'])
def logout():
    """Logout endpoint"""
    session.clear()
    logger.info("User logged out")
    return jsonify({'success': True, 'message': 'Logged out successfully'})

@app.route('/check-auth', methods=['GET'])
def check_auth():
    """Check if user is authenticated"""
    if 'logged_in' in session and session['logged_in']:
        return jsonify({
            'authenticated': True,
            'username': session.get('username', '')
        })
    return jsonify({'authenticated': False}), 401

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    """Main prediction endpoint"""
    # Handle OPTIONS request for CORS preflight
    if request.method == 'OPTIONS':
        return '', 200
    
    # Check authentication
    if 'logged_in' not in session or not session['logged_in']:
        return jsonify({'error': 'Authentication required'}), 401
    
    try:
        data = request.get_json()
        logger.info(f"Received prediction request: {data}")
        
        # Validate required fields
        required_fields = ['soil_type', 'lithology', 'land_use', 'latitude', 'longitude']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Set default values for optional fields
        defaults = {
            'rainfall_mm': 800,
            'slope_deg': 5.0,
            'elevation_m': 400,
            # water_table_m can be None/null, so we don't set a default if missing, 
            # but rely on the model or preprocess to handle it. 
            'distance_to_river_km': 2.0,
            'ndvi': 0.4
        }
        
        for key, default_value in defaults.items():
            # Check for explicit missing or None value
            if key not in data or data[key] is None:
                data[key] = default_value
        
        # Convert all to float/int where necessary, handling None for water_table_m
        data['latitude'] = float(data['latitude'])
        data['longitude'] = float(data['longitude'])
        data['rainfall_mm'] = float(data['rainfall_mm'])
        data['slope_deg'] = float(data['slope_deg'])
        data['elevation_m'] = float(data['elevation_m'])
        data['distance_to_river_km'] = float(data['distance_to_river_km'])
        data['ndvi'] = float(data['ndvi'])
        if 'water_table_m' in data and data['water_table_m'] is not None:
             data['water_table_m'] = float(data['water_table_m'])
        else:
             # Set to a median value for prediction if truly missing, but use the default logic
             data['water_table_m'] = 15.0 

        # Static/readymade override or missing models fallback
        if bool(data.get('use_static')) or not models or any(k not in models for k in ['suitability', 'depth', 'discharge', 'quality']):
            result = get_readymade_result(data)
            logger.info(f"Returning readymade result: {result}")
            return jsonify(result)

        # Preprocess input and run ML models
        X = preprocess_input(data)

        suitability_pred = models['suitability'].predict(X)[0]
        suitability_prob = models['suitability'].predict_proba(X)[0][1]
        depth_pred = models['depth'].predict(X)[0]
        discharge_pred = models['discharge'].predict(X)[0]
        quality_pred = models['quality'].predict(X)[0]

        result = {
            'suitable': int(suitability_pred),
            'suitable_probability': float(suitability_prob),
            'predicted_depth_m': float(depth_pred),
            'predicted_discharge_lps': float(discharge_pred),
            'predicted_quality_index': float(quality_pred),
            'input_data': data,
            'source': 'model'
        }

        logger.info(f"Prediction result: {result}")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        try:
            data = request.get_json() or {}
            # Ensure we have minimum required data for fallback
            if not data.get('land_use'):
                data['land_use'] = 'agriculture'
            result = get_readymade_result(data)
            result['warning'] = 'Using fallback prediction due to model error'
            logger.info(f"Returning readymade fallback result: {result}")
            return jsonify(result)
        except Exception as fallback_error:
            logger.error(f"Fallback also failed: {str(fallback_error)}")
            return jsonify({
                'error': 'Prediction failed',
                'message': str(e),
                'suitable': 0,
                'suitable_probability': 0.0,
                'predicted_depth_m': 0.0,
                'predicted_discharge_lps': 0.0,
                'predicted_quality_index': 0.0
            }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': len(models) > 0,
        'available_models': list(models.keys())
    })

@app.route('/', methods=['GET'])
def home():
    """Home endpoint - redirect based on authentication"""
    # If user is logged in, redirect to main app
    if 'logged_in' in session and session['logged_in']:
        username = session.get('username', '').lower()
        if username == 'admin':
            return redirect('/admin.html')
        return redirect('/index.html')
    
    # If not logged in, redirect to login
    return redirect('/home.html')

@app.route('/home.html', methods=['GET'])
def home_page():
    """Serve homepage"""
    # If user is logged in, redirect to main app
    if 'logged_in' in session and session['logged_in']:
        username = session.get('username', '').lower()
        if username == 'admin':
            return redirect('/admin.html')
        return redirect('/index.html')
    
    # Serve home page for non-logged in users
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        home_path = os.path.join(base_dir, 'home.html')
        with open(home_path, 'r', encoding='utf-8') as f:
            return f.read(), 200, {'Content-Type': 'text/html; charset=utf-8'}
    except Exception as e:
        logger.error(f"Error serving home.html: {e}")
        return redirect('/login.html')

@app.route('/about.html', methods=['GET'])
def about_page():
    """Serve about page"""
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        about_path = os.path.join(base_dir, 'about.html')
        with open(about_path, 'r', encoding='utf-8') as f:
            return f.read(), 200, {'Content-Type': 'text/html; charset=utf-8'}
    except Exception as e:
        logger.error(f"Error serving about.html: {e}")
        return f"Error loading about page: {str(e)}", 500

@app.route('/features.html', methods=['GET'])
def features_page():
    """Serve features page"""
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        features_path = os.path.join(base_dir, 'features.html')
        with open(features_path, 'r', encoding='utf-8') as f:
            return f.read(), 200, {'Content-Type': 'text/html; charset=utf-8'}
    except Exception as e:
        logger.error(f"Error serving features.html: {e}")
        return f"Error loading features page: {str(e)}", 500

@app.route('/login.html', methods=['GET'])
def login_page():
    """Serve login page"""
    # If user is already logged in, redirect to main app
    if 'logged_in' in session and session['logged_in']:
        username = session.get('username', '').lower()
        if username == 'admin':
            return redirect('/admin.html')
        return redirect('/index.html')
    
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        login_path = os.path.join(base_dir, 'login.html')
        with open(login_path, 'r', encoding='utf-8') as f:
            return f.read(), 200, {'Content-Type': 'text/html; charset=utf-8'}
    except Exception as e:
        logger.error(f"Error serving login.html: {e}")
        return f"Error loading login page: {str(e)}", 500

@app.route('/index.html', methods=['GET'])
def index():
    """Serve main application page"""
    # Redirect to login if not authenticated
    if 'logged_in' not in session or not session['logged_in']:
        return redirect('/login.html')
    
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        index_path = os.path.join(base_dir, 'index.html')
        with open(index_path, 'r', encoding='utf-8') as f:
            return f.read(), 200, {'Content-Type': 'text/html; charset=utf-8'}
    except Exception as e:
        logger.error(f"Error serving index.html: {e}")
        return f"Error loading index page: {str(e)}", 500

@app.route('/admin/users', methods=['GET'])
@admin_required
def admin_get_users():
    """Get all users (admin only)"""
    try:
        users = get_all_users()
        users_list = []
        for user in users:
            users_list.append({
                'id': user['id'],
                'username': user['username'],
                'email': user['email'] or '',
                'locality': user['locality'] or '',
                'created_at': user['created_at'] or '',
                'last_login': user['last_login'] or 'Never'
            })
        return jsonify({
            'success': True,
            'users': users_list,
            'total': len(users_list)
        })
    except Exception as e:
        logger.error(f"Error fetching users: {str(e)}", exc_info=True)
        return jsonify({'error': f'Failed to fetch users: {str(e)}'}), 500

@app.route('/admin.html', methods=['GET'])
def admin_page():
    """Serve admin page"""
    # Check if user is logged in and is admin
    if 'logged_in' not in session or not session['logged_in']:
        return redirect('/login.html')
    if session.get('username', '').lower() != 'admin':
        return '<h1>Access Denied</h1><p>Admin access required.</p>', 403
    
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        admin_path = os.path.join(base_dir, 'admin.html')
        with open(admin_path, 'r', encoding='utf-8') as f:
            return f.read(), 200, {'Content-Type': 'text/html; charset=utf-8'}
    except FileNotFoundError:
        # Create admin page if it doesn't exist
        return create_admin_page(), 200, {'Content-Type': 'text/html; charset=utf-8'}
    except Exception as e:
        logger.error(f"Error serving admin.html: {e}")
        return f"Error loading admin page: {str(e)}", 500

def create_profile_page():
    """Create profile page HTML"""
    # Returning the original content of profile.html if it was missing to ensure function completeness.
    # In a typical setup, we'd ensure profile.html exists, but since we have it, this function is mostly a fallback.
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>My Profile | AI Water Well Predictor</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        * { transition: all 0.3s ease; }
        body { 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            font-family: 'Inter', sans-serif;
            min-height: 100vh; 
            padding: 2rem; 
        }
        .profile-card { 
            background: white; 
            border-radius: 20px; 
            box-shadow: 0 20px 60px rgba(0,0,0,0.3); 
            padding: 2.5rem; 
            animation: fadeInUp 0.6s ease-out;
        }
        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(30px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .profile-picture {
            width: 150px;
            height: 150px;
            border-radius: 50%;
            object-fit: cover;
            border: 5px solid #667eea;
            cursor: pointer;
            transition: transform 0.3s ease;
        }
        .profile-picture:hover {
            transform: scale(1.05);
        }
        .form-control:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.1);
        }
        .btn-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: none;
            border-radius: 12px;
            padding: 0.75rem 1.5rem;
            font-weight: 600;
        }
        .btn-danger {
            background: linear-gradient(135deg, #dc3545 0%, #c82333 100%);
            border: none;
            border-radius: 12px;
            padding: 0.75rem 1.5rem;
            font-weight: 600;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="profile-card">
            <div class="d-flex justify-content-between align-items-center mb-4">
                <h1><i class="fas fa-user-circle text-primary"></i> My Profile</h1>
                <div>
                    <a href="/index.html" class="btn btn-outline-primary me-2"><i class="fas fa-home"></i> Home</a>
                    <button onclick="logout()" class="btn btn-outline-danger"><i class="fas fa-sign-out-alt"></i> Logout</button>
                </div>
            </div>
            
            <div id="alert-container"></div>
            
            <div class="row">
                <div class="col-md-4 text-center mb-4">
                    <div class="position-relative d-inline-block">
                        <img id="profile-picture" src="/static/profile_pics/default.png" alt="Profile Picture" class="profile-picture" onclick="document.getElementById('file-input').click()">
                        <input type="file" id="file-input" accept="image/*" style="display: none;" onchange="uploadProfilePicture(event)">
                        <div class="mt-3">
                            <button class="btn btn-sm btn-outline-primary" onclick="document.getElementById('file-input').click()">
                                <i class="fas fa-camera"></i> Change Picture
                            </button>
                        </div>
                    </div>
                </div>
                
                <div class="col-md-8">
                    <form id="profile-form">
                        <div class="mb-3">
                            <label class="form-label fw-semibold">Username</label>
                            <input type="text" id="username" class="form-control" readonly>
                        </div>
                        
                        <div class="mb-3">
                            <label class="form-label fw-semibold">Email</label>
                            <input type="email" id="email" class="form-control" placeholder="Enter your email">
                        </div>
                        
                        <div class="mb-3">
                            <label class="form-label fw-semibold">Locality</label>
                            <input type="text" id="locality" class="form-control" placeholder="Enter your locality">
                        </div>
                        
                        <div class="mb-3">
                            <label class="form-label fw-semibold">Account Created</label>
                            <input type="text" id="created-at" class="form-control" readonly>
                        </div>
                        
                        <div class="mb-3">
                            <label class="form-label fw-semibold">Last Login</label>
                            <input type="text" id="last-login" class="form-control" readonly>
                        </div>
                        
                        <button type="submit" class="btn btn-primary">
                            <i class="fas fa-save"></i> Save Changes
                        </button>
                    </form>
                </div>
            </div>
            
            <hr class="my-4">
            
            <div class="mt-4">
                <h5 class="text-danger mb-3"><i class="fas fa-exclamation-triangle"></i> Danger Zone</h5>
                <button class="btn btn-danger" data-bs-toggle="modal" data-bs-target="#deleteModal">
                    <i class="fas fa-trash-alt"></i> Delete Account
                </button>
            </div>
        </div>
    </div>
    
    <div class="modal fade" id="deleteModal" tabindex="-1">
        <div class="modal-dialog">
            <div class="modal-content">
                <div class="modal-header bg-danger text-white">
                    <h5 class="modal-title"><i class="fas fa-exclamation-triangle"></i> Delete Account</h5>
                    <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal"></button>
                </div>
                <div class="modal-body">
                    <p>Are you sure you want to delete your account? This action cannot be undone.</p>
                    <div class="mb-3">
                        <label class="form-label">Enter your password to confirm:</label>
                        <input type="password" id="delete-password" class="form-control" placeholder="Enter your password">
                    </div>
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
                    <button type="button" class="btn btn-danger" onclick="confirmDeleteAccount()">Delete Account</button>
                </div>
            </div>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        async function loadProfile() {
            try {
                const response = await fetch('/profile', { credentials: 'include' });
                if (response.status === 401) {
                    window.location.href = '/login.html';
                    return;
                }
                const data = await response.json();
                if (data.success) {
                    const profile = data.profile;
                    document.getElementById('username').value = profile.username;
                    document.getElementById('email').value = profile.email || '';
                    document.getElementById('locality').value = profile.locality || '';
                    document.getElementById('created-at').value = profile.created_at || 'N/A';
                    document.getElementById('last-login').value = profile.last_login || 'Never';
                    
                    if (profile.profile_picture) {
                        document.getElementById('profile-picture').src = profile.profile_picture;
                    } else {
                        document.getElementById('profile-picture').src = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTUwIiBoZWlnaHQ9IjE1MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTUwIiBoZWlnaHQ9IjE1MCIgZmlsbD0iI2U1ZTdlYiIvPjx0ZXh0IHg9IjUwJSIgeT0iNTAlIiBmb250LWZhbWlseT0iQXJpYWwiIGZvbnQtc2l6ZT0iNDgiIGZpbGw9IiM5Y2EzYWYiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGR5PSIuM2VtIj5Vc2VyPC90ZXh0Pjwvc3ZnPg==';
                    }
                }
            } catch (error) {
                console.error('Error loading profile:', error);
                showAlert('Error loading profile. Please refresh the page.', 'danger');
            }
        }
        
        async function uploadProfilePicture(event) {
            const file = event.target.files[0];
            if (!file) return;
            
            if (file.size > 5 * 1024 * 1024) {
                showAlert('File too large. Maximum size: 5MB', 'danger');
                return;
            }
            
            const formData = new FormData();
            formData.append('file', file);
            
            try {
                const response = await fetch('/profile/picture', {
                    method: 'POST',
                    credentials: 'include',
                    body: formData
                });
                
                const data = await response.json();
                if (data.success) {
                    document.getElementById('profile-picture').src = data.profile_picture;
                    showAlert('Profile picture updated successfully!', 'success');
                } else {
                    showAlert(data.error || 'Failed to upload picture', 'danger');
                }
            } catch (error) {
                console.error('Error uploading picture:', error);
                showAlert('Error uploading picture. Please try again.', 'danger');
            }
        }
        
        document.getElementById('profile-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const email = document.getElementById('email').value.trim();
            const locality = document.getElementById('locality').value.trim();
            
            try {
                const response = await fetch('/profile', {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    credentials: 'include',
                    body: JSON.stringify({ email, locality })
                });
                
                const data = await response.json();
                if (data.success) {
                    showAlert('Profile updated successfully!', 'success');
                } else {
                    showAlert(data.error || 'Failed to update profile', 'danger');
                }
            } catch (error) {
                console.error('Error updating profile:', error);
                showAlert('Error updating profile. Please try again.', 'danger');
            }
        });
        
        async function confirmDeleteAccount() {
            const password = document.getElementById('delete-password').value;
            if (!password) {
                showAlert('Please enter your password', 'danger');
                return;
            }
            
            if (!confirm('Are you absolutely sure? This cannot be undone!')) {
                return;
            }
            
            try {
                const response = await fetch('/profile', {
                    method: 'DELETE',
                    headers: { 'Content-Type': 'application/json' },
                    credentials: 'include',
                    body: JSON.stringify({ password })
                });
                
                const data = await response.json();
                if (data.success) {
                    showAlert('Account deleted successfully. Redirecting...', 'success');
                    setTimeout(() => {
                        window.location.href = '/login.html';
                    }, 2000);
                } else {
                    showAlert(data.error || 'Failed to delete account', 'danger');
                }
            } catch (error) {
                console.error('Error deleting account:', error);
                showAlert('Error deleting account. Please try again.', 'danger');
            }
        }
        
        function showAlert(message, type = 'danger') {
            const alertContainer = document.getElementById('alert-container');
            alertContainer.innerHTML = `
                <div class="alert alert-${type} alert-dismissible fade show" role="alert">
                    <i class="fas fa-${type === 'danger' ? 'exclamation-triangle' : 'check-circle'} me-2"></i>
                    ${message}
                    <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                </div>
            `;
        }
        
        async function logout() {
            try {
                await fetch('/logout', { method: 'POST', credentials: 'include' });
            } catch (error) {
                console.error('Logout error:', error);
            }
            window.location.href = '/login.html';
        }
        
        loadProfile();
    </script>
</body>
</html>'''

def create_admin_page():
    """Create admin page HTML"""
    # Returning the original content of admin.html if it was missing to ensure function completeness.
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Admin - User Management</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        body { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; padding: 2rem; }
        .admin-card { background: white; border-radius: 20px; box-shadow: 0 20px 60px rgba(0,0,0,0.3); padding: 2rem; }
        .table-responsive { max-height: 600px; overflow-y: auto; }
        .badge { padding: 0.5rem 1rem; }
    </style>
</head>
<body>
    <div class="container">
        <div class="admin-card">
            <div class="d-flex justify-content-between align-items-center mb-4">
                <h1><i class="fas fa-users-cog"></i> User Management</h1>
                <div>
                    <a href="/index.html" class="btn btn-outline-primary me-2"><i class="fas fa-home"></i> Home</a>
                    <button onclick="logout()" class="btn btn-outline-danger"><i class="fas fa-sign-out-alt"></i> Logout</button>
                </div>
            </div>
            <div class="mb-3">
                <button onclick="refreshUsers()" class="btn btn-primary"><i class="fas fa-sync-alt"></i> Refresh</button>
                <span class="ms-3" id="user-count"></span>
            </div>
            <div class="table-responsive">
                <table class="table table-striped table-hover">
                    <thead class="table-dark">
                        <tr>
                            <th>ID</th>
                            <th>Username</th>
                            <th>Email</th>
                            <th>Locality</th>
                            <th>Created At</th>
                            <th>Last Login</th>
                        </tr>
                    </thead>
                    <tbody id="users-table">
                        <tr><td colspan="6" class="text-center">Loading...</td></tr>
                    </tbody>
                </table>
            </div>
        </div>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        async function loadUsers() {
            try {
                const response = await fetch('/admin/users', {
                    credentials: 'include'
                });
                const data = await response.json();
                if (data.success) {
                    const tbody = document.getElementById('users-table');
                    const countSpan = document.getElementById('user-count');
                    countSpan.textContent = `Total Users: ${data.total}`;
                    
                    if (data.users.length === 0) {
                        tbody.innerHTML = '<tr><td colspan="6" class="text-center">No users found</td></tr>';
                        return;
                    }
                    
                    tbody.innerHTML = data.users.map(user => `
                        <tr>
                            <td>${user.id}</td>
                            <td><strong>${user.username}</strong></td>
                            <td>${user.email || '<span class="text-muted">N/A</span>'}</td>
                            <td>${user.locality || '<span class="text-muted">N/A</span>'}</td>
                            <td>${user.created_at || 'N/A'}</td>
                            <td>${user.last_login || '<span class="text-muted">Never</span>'}</td>
                        </tr>
                    `).join('');
                } else {
                    alert('Error: ' + (data.error || 'Failed to load users'));
                }
            } catch (error) {
                console.error('Error loading users:', error);
                document.getElementById('users-table').innerHTML = 
                    '<tr><td colspan="6" class="text-center text-danger">Error loading users. Please refresh.</td></tr>';
            }
        }
        
        function refreshUsers() {
            loadUsers();
        }
        
        async function logout() {
            try {
                await fetch('/logout', { method: 'POST', credentials: 'include' });
                window.location.href = '/login.html';
            } catch (error) {
                window.location.href = '/login.html';
            }
        }
        
        // Load users on page load
        loadUsers();
        // Auto-refresh every 30 seconds
        setInterval(loadUsers, 30000);
    </script>
</body>
</html>'''

@app.route('/profile', methods=['GET'])
@login_required
def get_profile():
    """Get current user's profile"""
    try:
        username = session.get('username')
        if not username:
            return jsonify({'error': 'Not authenticated'}), 401
        
        user = get_user(username)
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        profile_pic_url = None
        if user.get('profile_picture'):
            profile_pic_url = f'/static/profile_pics/{user["profile_picture"]}'
        
        return jsonify({
            'success': True,
            'profile': {
                'id': user['id'],
                'username': user['username'],
                'email': user['email'] or '',
                'locality': user['locality'] or '',
                'profile_picture': profile_pic_url,
                'created_at': user['created_at'] or '',
                'last_login': user['last_login'] or 'Never'
            }
        })
    except Exception as e:
        logger.error(f"Error fetching profile: {str(e)}", exc_info=True)
        return jsonify({'error': f'Failed to fetch profile: {str(e)}'}), 500

@app.route('/profile', methods=['PUT'])
@login_required
def update_profile():
    """Update user profile"""
    try:
        username = session.get('username')
        if not username:
            return jsonify({'error': 'Not authenticated'}), 401
        
        data = request.get_json()
        email = data.get('email', '').strip() if data.get('email') is not None else None
        locality = data.get('locality', '').strip() if data.get('locality') is not None else None
        
        # We need to explicitly check for None for fields that might be cleared
        # If the frontend sends an empty string, we treat it as None for database storage
        if email == '': email = None
        if locality == '': locality = None
        
        update_user_profile(username, email=email, locality=locality)
        
        user = get_user(username)
        profile_pic_url = None
        if user.get('profile_picture'):
            profile_pic_url = f'/static/profile_pics/{user["profile_picture"]}'
        
        return jsonify({
            'success': True,
            'message': 'Profile updated successfully',
            'profile': {
                'username': user['username'],
                'email': user['email'] or '',
                'locality': user['locality'] or '',
                'profile_picture': profile_pic_url
            }
        })
    except Exception as e:
        logger.error(f"Error updating profile: {str(e)}", exc_info=True)
        return jsonify({'error': f'Failed to update profile: {str(e)}'}), 500

@app.route('/profile/picture', methods=['POST'])
@login_required
def upload_profile_picture():
    """Upload profile picture"""
    try:
        username = session.get('username')
        if not username:
            return jsonify({'error': 'Not authenticated'}), 401
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed: PNG, JPG, JPEG, GIF, WEBP'}), 400
        
        # Check file size
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        
        if file_size > MAX_FILE_SIZE:
            return jsonify({'error': 'File too large. Maximum size: 5MB'}), 400
        
        # Get current user to delete old picture
        user = get_user(username)
        if user and user.get('profile_picture'):
            old_pic_path = os.path.join(PROFILE_PICS_DIR, user['profile_picture'])
            if os.path.exists(old_pic_path):
                try:
                    os.remove(old_pic_path)
                except Exception as e:
                    logger.error(f"Error deleting old profile picture: {e}")
        
        # Save new picture
        filename_parts = file.filename.rsplit('.', 1)
        extension = filename_parts[1].lower() if len(filename_parts) > 1 else 'png' # Default to png if no extension
        filename = secure_filename(f"{username}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{extension}")
        filepath = os.path.join(PROFILE_PICS_DIR, filename)
        file.save(filepath)
        
        # Update database
        update_user_profile(username, profile_picture=filename)
        
        return jsonify({
            'success': True,
            'message': 'Profile picture uploaded successfully',
            'profile_picture': f'/static/profile_pics/{filename}'
        })
    except Exception as e:
        logger.error(f"Error uploading profile picture: {str(e)}", exc_info=True)
        return jsonify({'error': f'Failed to upload picture: {str(e)}'}), 500

@app.route('/profile', methods=['DELETE'])
@login_required
def delete_account():
    """Delete user account"""
    try:
        username = session.get('username')
        if not username:
            return jsonify({'error': 'Not authenticated'}), 401
        
        # Prevent admin from deleting their own account
        if username.lower() == 'admin':
            return jsonify({'error': 'Cannot delete admin account'}), 403
        
        data = request.get_json()
        password = data.get('password', '') if data else ''
        
        if not password:
            return jsonify({'error': 'Password required to delete account'}), 400
        
        # Verify password
        user = get_user(username)
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        if not check_password_hash(user['password_hash'], password):
            return jsonify({'error': 'Incorrect password'}), 401
        
        # Delete account
        if delete_user_account(username):
            session.clear()
            return jsonify({
                'success': True,
                'message': 'Account deleted successfully'
            })
        else:
            return jsonify({'error': 'Failed to delete account'}), 500
            
    except Exception as e:
        logger.error(f"Error deleting account: {str(e)}", exc_info=True)
        return jsonify({'error': f'Failed to delete account: {str(e)}'}), 500

@app.route('/static/profile_pics/<filename>')
def serve_profile_picture(filename):
    """Serve profile pictures"""
    # Check for authentication if needed, but for simple serving we let it go
    return send_from_directory(PROFILE_PICS_DIR, filename)

@app.route('/profile.html', methods=['GET'])
def profile_page():
    """Serve profile page"""
    # Check if user is logged in - redirect to login if not
    if 'logged_in' not in session or not session['logged_in']:
        return redirect('/login.html')
    
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        profile_path = os.path.join(base_dir, 'profile.html')
        with open(profile_path, 'r', encoding='utf-8') as f:
            return f.read(), 200, {'Content-Type': 'text/html; charset=utf-8'}
    except FileNotFoundError:
        # Create profile page if it doesn't exist
        return create_profile_page(), 200, {'Content-Type': 'text/html; charset=utf-8'}
    except Exception as e:
        logger.error(f"Error serving profile.html: {e}")
        return f"Error loading profile page: {str(e)}", 500

@app.route('/api', methods=['GET'])
def api_info():
    """API information endpoint"""
    return jsonify({
        'message': 'AI Water Well Predictor API',
        'version': '1.0.0',
        'endpoints': {
            'POST /signup': 'User signup/registration',
            'POST /login': 'User login',
            'POST /logout': 'User logout',
            'GET /check-auth': 'Check authentication status',
            'GET /admin/users': 'Get all users (admin only)',
            'GET /admin.html': 'Admin user management page (admin only)',
            'POST /predict': 'Make water well predictions (requires auth)',
            'GET /health': 'Health check',
            'GET /api': 'This information'
        },
        'required_fields': ['soil_type', 'lithology', 'land_use', 'latitude', 'longitude'],
        'optional_fields': ['rainfall_mm', 'slope_deg', 'elevation_m', 'water_table_m', 'distance_to_river_km', 'ndvi']
    })

if __name__ == '__main__':
    # Initialize database on startup
    init_db()
    # Train models on startup
    train_models()
    
    # Run the app on the default local port
    port = int(os.environ.get('PORT', 8000))
    # Running the app with debug=True allows hot-reloading in VS Code
    app.run(host='0.0.0.0', port=port, debug=True)