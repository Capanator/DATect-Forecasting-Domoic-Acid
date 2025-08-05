"""
Model definitions and training logic without data leakage.
"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.base import clone


class ModelFactory:
    """Factory for creating different model types."""
    
    @staticmethod
    def create_regression_models(random_state=42):
        """Create regression models."""
        return {
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                random_state=random_state,
                n_jobs=1
            ),
            'linear': LinearRegression(),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                random_state=random_state
            )
        }
    
    @staticmethod
    def create_classification_models(random_state=42):
        """Create classification models."""
        return {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                random_state=random_state,
                n_jobs=1
            ),
            'logistic': LogisticRegression(
                solver='lbfgs',
                max_iter=1000,
                random_state=random_state
            )
        }
    
    @staticmethod
    def create_quantile_models(quantiles=[0.05, 0.5, 0.95], random_state=42):
        """Create quantile regression models."""
        models = {}
        for q in quantiles:
            models[f'q{int(q*100):02d}'] = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                loss='quantile',
                alpha=q,
                random_state=random_state
            )
        return models


class ModelTrainer:
    """Handles model training with proper time series validation."""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.trained_models_ = {}
        self.best_params_ = {}
    
    def create_preprocessing_pipeline(self):
        """Create preprocessing pipeline."""
        numeric_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', MinMaxScaler())
        ])
        
        preprocessor = ColumnTransformer([
            ('num', numeric_pipeline, 'passthrough')
        ], remainder='drop')
        
        return preprocessor
    
    def grid_search_model(self, model, param_grid, X_train, y_train, 
                         scoring='r2', cv_splits=5):
        """Perform grid search with time series cross-validation."""
        preprocessor = self.create_preprocessing_pipeline()
        
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        # Use TimeSeriesSplit for proper time series validation
        tscv = TimeSeriesSplit(n_splits=cv_splits)
        
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            scoring=scoring,
            cv=tscv,
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        
        return grid_search.best_estimator_, grid_search.best_params_
    
    def train_single_model(self, model, X_train, y_train):
        """Train a single model with preprocessing."""
        preprocessor = self.create_preprocessing_pipeline()
        
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        pipeline.fit(X_train, y_train)
        return pipeline
    
    def train_quantile_models(self, X_train, y_train, quantiles=[0.05, 0.5, 0.95]):
        """Train multiple quantile regression models."""
        models = ModelFactory.create_quantile_models(quantiles, self.random_state)
        trained_models = {}
        
        for name, model in models.items():
            trained_models[name] = self.train_single_model(model, X_train, y_train)
        
        return trained_models


class ModelPredictor:
    """Handles predictions from trained models."""
    
    def __init__(self):
        pass
    
    def predict_regression(self, model, X_test):
        """Make regression predictions."""
        try:
            predictions = model.predict(X_test)
            return predictions
        except Exception as e:
            print(f"Error in regression prediction: {e}")
            return np.full(len(X_test), np.nan)
    
    def predict_classification(self, model, X_test):
        """Make classification predictions with probabilities."""
        try:
            predictions = model.predict(X_test)
            probabilities = model.predict_proba(X_test)
            return predictions, probabilities
        except Exception as e:
            print(f"Error in classification prediction: {e}")
            n_classes = len(model.classes_) if hasattr(model, 'classes_') else 4
            return (np.full(len(X_test), np.nan), 
                   np.full((len(X_test), n_classes), np.nan))
    
    def predict_quantiles(self, models, X_test):
        """Make quantile predictions."""
        predictions = {}
        for name, model in models.items():
            try:
                predictions[name] = model.predict(X_test)
            except Exception as e:
                print(f"Error in quantile prediction for {name}: {e}")
                predictions[name] = np.full(len(X_test), np.nan)
        
        return predictions