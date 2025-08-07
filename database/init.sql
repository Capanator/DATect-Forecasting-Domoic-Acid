-- DATect Database Initialization
-- ================================
-- Initial schema for user management and forecast storage (future features)

-- Create extension for UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Users table (for future authentication)
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    full_name VARCHAR(255),
    is_active BOOLEAN DEFAULT TRUE,
    is_admin BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Forecast results table (for caching and history)
CREATE TABLE IF NOT EXISTS forecast_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id),
    site VARCHAR(100) NOT NULL,
    forecast_date DATE NOT NULL,
    model_type VARCHAR(50) NOT NULL,
    task_type VARCHAR(20) NOT NULL CHECK (task_type IN ('regression', 'classification')),
    prediction_value DECIMAL(10, 4),
    predicted_category INTEGER,
    training_samples INTEGER,
    feature_importance JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Index for common queries
    UNIQUE(site, forecast_date, model_type, task_type, user_id)
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_forecast_results_site_date ON forecast_results(site, forecast_date);
CREATE INDEX IF NOT EXISTS idx_forecast_results_user_created ON forecast_results(user_id, created_at);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);

-- Insert a default admin user (password: 'admin' - change in production!)
INSERT INTO users (email, hashed_password, full_name, is_admin) 
VALUES (
    'admin@datect.org', 
    '$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW', -- 'admin' hashed
    'System Administrator', 
    TRUE
) ON CONFLICT (email) DO NOTHING;