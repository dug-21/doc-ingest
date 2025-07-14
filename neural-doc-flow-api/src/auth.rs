//! Authentication and authorization management

use std::sync::Arc;
use sqlx::{Pool, Sqlite};
use jsonwebtoken::{encode, decode, Header, Algorithm, Validation, EncodingKey, DecodingKey};
use serde::{Deserialize, Serialize};
use argon2::{Argon2, PasswordHash, PasswordHasher, PasswordVerifier};
use argon2::password_hash::{rand_core::OsRng, SaltString};

use crate::error::{ApiError, ApiResult};
use crate::models::{LoginRequest, LoginResponse, RegisterRequest, UserInfo};

/// JWT claims structure
#[derive(Debug, Serialize, Deserialize)]
pub struct Claims {
    pub sub: String,  // Subject (user ID)
    pub username: String,
    pub role: String,
    pub exp: usize,   // Expiration time
    pub iat: usize,   // Issued at
}

/// Authentication manager
pub struct AuthManager {
    jwt_secret: String,
    db: Pool<Sqlite>,
    encoding_key: EncodingKey,
    decoding_key: DecodingKey,
    validation: Validation,
}

impl AuthManager {
    pub async fn new(jwt_secret: String, db: Pool<Sqlite>) -> ApiResult<Self> {
        // Initialize database tables
        sqlx::query(r#"
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                full_name TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_login TEXT,
                is_active BOOLEAN NOT NULL DEFAULT 1
            )
        "#)
        .execute(&db)
        .await?;

        let encoding_key = EncodingKey::from_secret(jwt_secret.as_ref());
        let decoding_key = DecodingKey::from_secret(jwt_secret.as_ref());
        let mut validation = Validation::new(Algorithm::HS256);
        validation.set_audience(&["neural-doc-flow-api"]);

        Ok(Self {
            jwt_secret,
            db,
            encoding_key,
            decoding_key,
            validation,
        })
    }

    /// Authenticate user and return JWT token
    pub async fn login(&self, request: LoginRequest) -> ApiResult<LoginResponse> {
        // Find user by username or email
        let user = sqlx::query!(
            "SELECT id, username, email, password_hash, full_name, role FROM users WHERE username = ? OR email = ? AND is_active = 1",
            request.username, request.username
        )
        .fetch_optional(&self.db)
        .await?;

        let user = user.ok_or_else(|| ApiError::Unauthorized {
            message: "Invalid credentials".to_string(),
        })?;

        // Verify password
        let argon2 = Argon2::default();
        let parsed_hash = PasswordHash::new(&user.password_hash)
            .map_err(|_| ApiError::InternalError {
                message: "Invalid password hash in database".to_string(),
            })?;

        argon2.verify_password(request.password.as_bytes(), &parsed_hash)
            .map_err(|_| ApiError::Unauthorized {
                message: "Invalid credentials".to_string(),
            })?;

        // Update last login
        sqlx::query!(
            "UPDATE users SET last_login = datetime('now') WHERE id = ?",
            user.id
        )
        .execute(&self.db)
        .await?;

        // Generate JWT token
        let now = chrono::Utc::now();
        let exp = now + chrono::Duration::hours(24); // 24 hour expiration

        let claims = Claims {
            sub: user.id.clone(),
            username: user.username.clone(),
            role: user.role.clone(),
            exp: exp.timestamp() as usize,
            iat: now.timestamp() as usize,
        };

        let token = encode(&Header::default(), &claims, &self.encoding_key)
            .map_err(|e| ApiError::InternalError {
                message: format!("Failed to generate token: {}", e),
            })?;

        Ok(LoginResponse {
            access_token: token,
            token_type: "Bearer".to_string(),
            expires_at: exp,
            user: UserInfo {
                id: user.id,
                username: user.username,
                email: user.email,
                full_name: user.full_name,
                role: user.role,
                created_at: chrono::Utc::now(), // TODO: get from DB
                last_login: Some(chrono::Utc::now()),
            },
        })
    }

    /// Register new user
    pub async fn register(&self, request: RegisterRequest) -> ApiResult<UserInfo> {
        // Check if username or email already exists
        let existing = sqlx::query!(
            "SELECT id FROM users WHERE username = ? OR email = ?",
            request.username, request.email
        )
        .fetch_optional(&self.db)
        .await?;

        if existing.is_some() {
            return Err(ApiError::BadRequest {
                message: "Username or email already exists".to_string(),
            });
        }

        // Hash password
        let argon2 = Argon2::default();
        let salt = SaltString::generate(&mut OsRng);
        let password_hash = argon2.hash_password(request.password.as_bytes(), &salt)
            .map_err(|e| ApiError::InternalError {
                message: format!("Failed to hash password: {}", e),
            })?
            .to_string();

        // Insert user
        let user_id = uuid::Uuid::new_v4().to_string();
        let now = chrono::Utc::now().to_rfc3339();

        sqlx::query!(
            r#"
            INSERT INTO users (id, username, email, password_hash, full_name, role, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, 'user', ?, ?)
            "#,
            user_id, request.username, request.email, password_hash, request.full_name, now, now
        )
        .execute(&self.db)
        .await?;

        Ok(UserInfo {
            id: user_id,
            username: request.username,
            email: request.email,
            full_name: request.full_name,
            role: "user".to_string(),
            created_at: chrono::Utc::now(),
            last_login: None,
        })
    }

    /// Validate JWT token and return claims
    pub fn validate_token(&self, token: &str) -> ApiResult<Claims> {
        let token_data = decode::<Claims>(token, &self.decoding_key, &self.validation)
            .map_err(|e| ApiError::Unauthorized {
                message: format!("Invalid token: {}", e),
            })?;

        Ok(token_data.claims)
    }

    /// Get user by ID
    pub async fn get_user(&self, user_id: &str) -> ApiResult<UserInfo> {
        let user = sqlx::query!(
            "SELECT id, username, email, full_name, role, created_at, last_login FROM users WHERE id = ? AND is_active = 1",
            user_id
        )
        .fetch_optional(&self.db)
        .await?;

        let user = user.ok_or_else(|| ApiError::NotFound {
            message: "User not found".to_string(),
        })?;

        Ok(UserInfo {
            id: user.id,
            username: user.username,
            email: user.email,
            full_name: user.full_name,
            role: user.role,
            created_at: chrono::DateTime::parse_from_rfc3339(&user.created_at)
                .unwrap_or_default()
                .with_timezone(&chrono::Utc),
            last_login: user.last_login.and_then(|login| {
                chrono::DateTime::parse_from_rfc3339(&login).ok()
            }).map(|dt| dt.with_timezone(&chrono::Utc)),
        })
    }

    /// Check if user has required role
    pub fn check_role(&self, claims: &Claims, required_role: &str) -> ApiResult<()> {
        // Simple role hierarchy: admin > user
        let has_permission = match (claims.role.as_str(), required_role) {
            ("admin", _) => true,
            ("user", "user") => true,
            _ => false,
        };

        if has_permission {
            Ok(())
        } else {
            Err(ApiError::Forbidden {
                message: format!("Required role: {}", required_role),
            })
        }
    }

    /// Extract token from Authorization header
    pub fn extract_token(auth_header: &str) -> Option<&str> {
        if auth_header.starts_with("Bearer ") {
            Some(&auth_header[7..])
        } else {
            None
        }
    }
}