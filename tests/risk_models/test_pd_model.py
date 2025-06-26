import unittest
from unittest.mock import MagicMock, patch, call
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import joblib
import datetime

from src.risk_models.pd_model import PDModel
from src.data_management.knowledge_base import KnowledgeBaseService
from src.data_management.ontology import (
    LoanAgreement, CorporateEntity, FinancialStatement,
    Currency, IndustrySector, CollateralType # Removed SeniorityOfDebt, Changed CurrencyCode
)
from src.core.config import settings
from src.mlops.model_registry import ModelRegistry


class TestPDModel(unittest.TestCase):

    def setUp(self):
        self.temp_model_dir = tempfile.TemporaryDirectory()
        self.test_model_path = Path(self.temp_model_dir.name) / "test_pd_model.joblib"
        self.pd_model = PDModel(model_path=self.test_model_path)

        self.mock_kb_service = MagicMock(spec=KnowledgeBaseService)

        self.mock_kb_service = MagicMock(spec=KnowledgeBaseService)

        # Sample data for mocking KB
        self.company1 = CorporateEntity(
            company_id="C1", company_name="TestCorp", industry_sector=IndustrySector.TECHNOLOGY,
            country_iso_code="US", founded_date=datetime.date(2000, 1, 1),
            # financial_statement_ids=["FS1"] # Not directly used by PD model's _prepare_features
        )
        self.company2 = CorporateEntity(
            company_id="C2", company_name="AnotherCorp", industry_sector=IndustrySector.FINANCIAL_SERVICES,
            country_iso_code="GB", founded_date=datetime.date(2010, 6, 15)
        )
        self.fs1_c1 = FinancialStatement( # Ensure all required FS fields are present
            statement_id="FS1", company_id="C1", statement_date=datetime.date(2022, 12, 31),
            currency=Currency.USD, revenue=1000000, net_income=100000,
            total_assets_usd=500000, total_liabilities_usd=200000, net_equity_usd=300000,
            reporting_period_months=12, # Added
            current_assets=150000, current_liabilities=50000
        )
        # FS for C2 to test different ratios
        self.fs1_c2 = FinancialStatement( # Ensure all required FS fields are present
            statement_id="FS2", company_id="C2", statement_date=datetime.date(2023, 3, 31),
            currency=Currency.GBP,
            revenue=500000, net_income=20000,
            total_assets_usd=250000, total_liabilities_usd=150000, net_equity_usd=100000,
            reporting_period_months=12, # Added
            current_assets=80000, current_liabilities=60000
        )


        self.loan1_c1 = LoanAgreement(
            loan_id="L1", company_id="C1", loan_amount=100000, currency=Currency.USD,
            origination_date=datetime.date(2022, 1, 1), maturity_date=datetime.date(2024, 1, 1),
            interest_rate_percentage=5.0, collateral_type=CollateralType.NONE,
            seniority_of_debt="Senior", default_status=False
        )
        self.loan2_c1 = LoanAgreement( # Another loan for C1, defaulted
            loan_id="L2", company_id="C1", loan_amount=50000, currency=Currency.USD,
            origination_date=datetime.date(2023, 1, 1), maturity_date=datetime.date(2023, 6, 1), # Short term
            interest_rate_percentage=8.0, collateral_type=CollateralType.INVENTORY,
            seniority_of_debt="Junior", default_status=True
        )
        self.loan1_c2 = LoanAgreement( # Loan for C2
            loan_id="L3", company_id="C2", loan_amount=200000, currency=Currency.GBP,
            origination_date=datetime.date(2023, 3, 1), maturity_date=datetime.date(2026, 3, 1),
            interest_rate_percentage=3.5, collateral_type=CollateralType.REAL_ESTATE,
            seniority_of_debt="Senior", default_status=False
        )

        self.mock_kb_service.get_all_companies.return_value = [self.company1, self.company2]
        self.mock_kb_service.get_all_loans.return_value = [self.loan1_c1, self.loan2_c1, self.loan1_c2]

        def mock_get_fs(company_id):
            if company_id == "C1": return [self.fs1_c1]
            if company_id == "C2": return [self.fs1_c2]
            return []
        self.mock_kb_service.get_financial_statements_for_company.side_effect = mock_get_fs


    def tearDown(self):
        self.temp_model_dir.cleanup()

    def test_prepare_features_structure_and_values(self):
        """Test the _prepare_features method for correct DataFrame structure and some values."""
        features_df = self.pd_model._prepare_features(self.mock_kb_service)

        self.assertIsInstance(features_df, pd.DataFrame)
        self.assertEqual(len(features_df), 3) # For L1, L2, L3

        expected_cols = [
            'loan_id', 'company_id', 'loan_amount_usd', 'interest_rate_percentage',
            'collateral_type', 'industry_sector', 'loan_duration_days',
            'company_age_at_origination', 'debt_to_equity_ratio', 'current_ratio',
            'net_profit_margin', 'roe', 'loan_amount_x_interest_rate', 'default_status'
        ]
        for col in expected_cols:
            self.assertIn(col, features_df.columns)

        # Check some calculated values for loan L1 (company C1, fs FS1_C1)
        l1_features = features_df[features_df['loan_id'] == 'L1'].iloc[0]
        self.assertEqual(l1_features['loan_duration_days'], (self.loan1_c1.maturity_date - self.loan1_c1.origination_date).days)
        expected_age_l1 = (self.loan1_c1.origination_date - self.company1.founded_date).days / 365.25
        self.assertAlmostEqual(l1_features['company_age_at_origination'], expected_age_l1)
        self.assertAlmostEqual(l1_features['debt_to_equity_ratio'], self.fs1_c1.total_liabilities_usd / self.fs1_c1.net_equity_usd)
        self.assertAlmostEqual(l1_features['current_ratio'], self.fs1_c1.current_assets / self.fs1_c1.current_liabilities)
        self.assertAlmostEqual(l1_features['net_profit_margin'], self.fs1_c1.net_income / self.fs1_c1.revenue)
        self.assertAlmostEqual(l1_features['roe'], self.fs1_c1.net_income / self.fs1_c1.net_equity_usd)
        self.assertEqual(l1_features['loan_amount_x_interest_rate'], self.loan1_c1.loan_amount * self.loan1_c1.interest_rate_percentage)
        self.assertEqual(l1_features['default_status'], 0)

        # Check defaulted loan L2
        l2_features = features_df[features_df['loan_id'] == 'L2'].iloc[0]
        self.assertEqual(l2_features['default_status'], 1)
        self.assertEqual(l2_features['collateral_type'], self.loan2_c1.collateral_type.value)
        self.assertEqual(l2_features['industry_sector'], self.company1.industry_sector.value)


    def test_prepare_features_missing_fs(self):
        """Test _prepare_features when a company has no financial statements."""
        self.mock_kb_service.get_financial_statements_for_company.return_value = [] # No FS for anyone
        features_df = self.pd_model._prepare_features(self.mock_kb_service)
        financial_ratios = ['debt_to_equity_ratio', 'current_ratio', 'net_profit_margin', 'roe']
        for ratio in financial_ratios:
            self.assertTrue(features_df[ratio].isnull().all())

    def test_prepare_features_missing_dates(self):
        """Test _prepare_features with missing company founded_date or loan origination_date."""
        self.company1.founded_date = None # Remove founded date for C1
        self.loan1_c1.maturity_date = None # Remove maturity for L1
        self.mock_kb_service.get_all_companies.return_value = [self.company1, self.company2]
        self.mock_kb_service.get_all_loans.return_value = [self.loan1_c1, self.loan2_c1, self.loan1_c2]

        features_df = self.pd_model._prepare_features(self.mock_kb_service)
        l1_features = features_df[features_df['loan_id'] == 'L1'].iloc[0]

        self.assertEqual(l1_features['company_age_at_origination'], -1) # Due to missing founded_date
        self.assertEqual(l1_features['loan_duration_days'], -1) # Due to missing maturity_date


    @patch('src.risk_models.pd_model.ModelRegistry')
    @patch('src.risk_models.pd_model.joblib.dump')
    def test_train_successful(self, mock_joblib_dump, MockModelRegistry):
        """Test successful model training and registration."""
        mock_registry_instance = MockModelRegistry.return_value

        # Make sure there's at least one default and one non-default for stratification
        self.loan1_c1.default_status = False
        self.loan2_c1.default_status = True # Already true, but explicit
        self.loan1_c2.default_status = False
        self.mock_kb_service.get_all_loans.return_value = [self.loan1_c1, self.loan2_c1, self.loan1_c2]


        metrics = self.pd_model.train(self.mock_kb_service)

        self.assertIn('train_accuracy', metrics)
        self.assertIn('test_accuracy', metrics)
        self.assertIn('test_roc_auc', metrics)
        self.assertNotIn('error', metrics)
        self.assertIsNotNone(self.pd_model.model)
        mock_joblib_dump.assert_called_once_with(self.pd_model.model, self.test_model_path)

        # Check model registration call
        mock_registry_instance.register_model.assert_called_once()
        args, kwargs = mock_registry_instance.register_model.call_args
        self.assertEqual(kwargs['model_name'], "PDModel")
        self.assertEqual(kwargs['model_path'], str(self.test_model_path))
        self.assertEqual(kwargs['metrics'], metrics)
        self.assertIn('n_estimators', kwargs['parameters']) # Check if some RF params are there

    def test_train_insufficient_data(self):
        """Test training with insufficient data."""
        self.mock_kb_service.get_all_loans.return_value = [self.loan1_c1] # Only one loan
        metrics = self.pd_model.train(self.mock_kb_service)
        self.assertIn('error', metrics)
        self.assertIn("Insufficient data", metrics['error'])
        self.assertIsNone(self.pd_model.model)

    def test_train_only_one_class(self):
        """Test training when data results in only one class in training set."""
        # All loans non-defaulted
        self.loan1_c1.default_status = False
        self.loan2_c1.default_status = False
        self.loan1_c2.default_status = False
        self.mock_kb_service.get_all_loans.return_value = [self.loan1_c1, self.loan2_c1, self.loan1_c2]

        metrics = self.pd_model.train(self.mock_kb_service, test_size=0.2) # Small test size to make it likely
        self.assertIn('error', metrics)
        self.assertTrue("one class" in metrics['error'].lower())
        self.assertIsNone(self.pd_model.model)


    def test_predict_and_predict_for_loan(self):
        """Test prediction methods after training a model."""
        # Train a model first (simplified version)
        self.loan1_c1.default_status = False
        self.loan2_c1.default_status = True
        self.loan1_c2.default_status = False
        self.mock_kb_service.get_all_loans.return_value = [self.loan1_c1, self.loan2_c1, self.loan1_c2]
        with patch('src.risk_models.pd_model.ModelRegistry'), patch('src.risk_models.pd_model.joblib.dump'):
            self.pd_model.train(self.mock_kb_service)

        self.assertIsNotNone(self.pd_model.model)

        # Prepare data for batch predict (should match columns from _prepare_features, before target drop)
        # We use the output of _prepare_features directly for testing predict()
        raw_features_df = self.pd_model._prepare_features(self.mock_kb_service)
        predict_df_input = raw_features_df.drop(['default_status', 'loan_id', 'company_id'], axis=1).head(2)

        batch_predictions = self.pd_model.predict(predict_df_input)
        self.assertIsInstance(batch_predictions, pd.DataFrame)
        self.assertEqual(len(batch_predictions), 2)
        self.assertIn('pd_prediction', batch_predictions.columns)
        self.assertIn('pd_probability', batch_predictions.columns)

        # Test predict_for_loan
        single_loan_data = self.loan1_c1.model_dump()
        single_company_data = self.company1.model_dump()
        # Add financial data to company_data dict if predict_for_loan expects it implicitly
        # (Current predict_for_loan doesn't fetch FS, so ratios will be NaN)

        prediction_tuple = self.pd_model.predict_for_loan(single_loan_data, single_company_data)
        self.assertIsNotNone(prediction_tuple)
        self.assertIsInstance(prediction_tuple[0], int) # Prediction class
        self.assertIsInstance(prediction_tuple[1], float) # Probability

    def test_predict_no_model(self):
        """Test prediction when model is not trained/loaded."""
        self.assertIsNone(self.pd_model.model)
        sample_df = pd.DataFrame([{'loan_amount_usd': 100}]) # Dummy df
        self.assertIsNone(self.pd_model.predict(sample_df))
        self.assertIsNone(self.pd_model.predict_for_loan({}, {}))


    def test_save_and_load_model(self):
        """Test saving and loading the model."""
        # Train a dummy model
        df = pd.DataFrame({
            'loan_amount_usd': [100, 200, 150, 300, 250],
            'interest_rate_percentage': [5, 6, 5.5, 7, 6.5],
            'collateral_type': ['None', 'Low', 'Mid', 'High', 'None'],
            'industry_sector': ['Tech', 'Finance', 'Tech', 'Retail', 'Finance'],
            'loan_duration_days': [365, 730, 365, 1095, 730],
            'company_age_at_origination': [5, 10, 3, 15, 8],
            'debt_to_equity_ratio': [0.5, 1.0, 0.3, 1.5, 0.8],
            'current_ratio': [1.5, 2.0, 1.2, 2.5, 1.8],
            'net_profit_margin': [0.1, 0.05, 0.12, 0.08, 0.09],
            'roe': [0.15, 0.1, 0.18, 0.12, 0.14],
            'loan_amount_x_interest_rate': [500,1200,825,2100,1625],
            'default_status': [0, 1, 0, 1, 0]
        })
        X = df.drop('default_status', axis=1)
        y = df['default_status']

        # Temporarily set numerical and categorical features for this dummy model
        original_num_feats = self.pd_model.numerical_features
        original_cat_feats = self.pd_model.categorical_features
        self.pd_model.numerical_features = [
            'loan_amount_usd', 'interest_rate_percentage', 'loan_duration_days',
            'company_age_at_origination', 'debt_to_equity_ratio', 'current_ratio',
            'net_profit_margin', 'roe', 'loan_amount_x_interest_rate'
        ]
        self.pd_model.categorical_features = ['collateral_type', 'industry_sector']

        # Rebuild the preprocessor part of the base_model with current feature lists
        from sklearn.compose import ColumnTransformer # Import if not already at top
        from sklearn.pipeline import Pipeline as SklearnPipeline # Alias to avoid conflict if Pipeline is defined elsewhere
        from sklearn.impute import SimpleImputer # Import if not already at top
        from sklearn.preprocessing import StandardScaler, OneHotEncoder # Import if not already at top

        classifier_step = self.pd_model.base_model.steps[1] # ('classifier', RandomForestClassifier(...))

        current_num_feats = self.pd_model.numerical_features
        current_cat_feats = self.pd_model.categorical_features

        new_preprocessor = ColumnTransformer(
            transformers=[
                ('num', SklearnPipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), current_num_feats),
                ('cat', SklearnPipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), current_cat_feats)
            ]
        )
        self.pd_model.base_model = SklearnPipeline(steps=[('preprocessor', new_preprocessor), classifier_step])

        self.pd_model.model = self.pd_model.base_model.fit(X, y)
        self.pd_model._feature_names = self.pd_model.model.named_steps['preprocessor'].get_feature_names_out()


        self.pd_model.save_model()
        self.assertTrue(self.test_model_path.exists())

        new_pd_model = PDModel(model_path=self.test_model_path)
        load_success = new_pd_model.load_model()
        self.assertTrue(load_success)
        self.assertIsNotNone(new_pd_model.model)

        # Restore original features
        self.pd_model.numerical_features = original_num_feats
        self.pd_model.categorical_features = original_cat_feats


    def test_load_model_not_exists(self):
        """Test loading a model that does not exist (and no registry fallback)."""
        non_existent_path = Path(self.temp_model_dir.name) / "ghost_model.joblib"
        new_pd_model = PDModel(model_path=non_existent_path)
        with patch('src.risk_models.pd_model.ModelRegistry') as MockReg:
            MockReg.return_value.get_production_model_path.return_value = None
            load_success = new_pd_model.load_model()
        self.assertFalse(load_success)
        self.assertIsNone(new_pd_model.model)

    @patch('src.risk_models.pd_model.joblib.load')
    @patch('src.risk_models.pd_model.ModelRegistry')
    def test_load_model_fallback_to_registry(self, MockModelRegistry, mock_joblib_load):
        """Test loading model falls back to registry if primary path fails."""
        non_existent_path = Path(self.temp_model_dir.name) / "ghost_model.joblib"
        registry_model_path_str = str(Path(self.temp_model_dir.name) / "registry_pd_model.joblib")

        # Create a simple, real scikit-learn pipeline to dump for the registry model
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler # Ensure this is imported if not already
        dummy_skl_pipeline = Pipeline([('scaler', StandardScaler())])
        # Create some dummy data to fit it so it's a valid "trained" model
        dummy_X = pd.DataFrame({'feature': [1,2,3]})
        dummy_skl_pipeline.fit(dummy_X)

        with open(registry_model_path_str, 'wb') as f:
            joblib.dump(dummy_skl_pipeline, f)

        mock_registry_instance = MockModelRegistry.return_value
        mock_registry_instance.get_production_model_path.return_value = registry_model_path_str

        mock_joblib_load.return_value = dummy_skl_pipeline # joblib.load will return this object

        new_pd_model = PDModel(model_path=non_existent_path)
        load_success = new_pd_model.load_model()

        self.assertTrue(load_success)
        self.assertIsNotNone(new_pd_model.model)
        mock_registry_instance.get_production_model_path.assert_called_once_with("PDModel")
        # joblib.load should be called twice if primary path fails then registry path is tried
        # Once for non_existent_path (fails internally), then for registry_model_path_str
        # However, our joblib.load is mocked globally.
        # The first call to joblib.load (for non_existent_path) will raise FileNotFoundError.
        # The PDModel.load_model catches this and then tries registry.
        # So, the mocked joblib.load is effectively only called for the registry path here.
        mock_joblib_load.assert_called_once_with(Path(registry_model_path_str))
        self.assertEqual(new_pd_model.model_path, Path(registry_model_path_str))


    @patch('src.risk_models.pd_model.shap.TreeExplainer')
    def test_get_feature_importance_shap(self, MockTreeExplainer):
        """Test SHAP feature importance calculation."""
        # Train a model (simplified)
        self.loan1_c1.default_status = False
        self.loan2_c1.default_status = True
        self.mock_kb_service.get_all_loans.return_value = [self.loan1_c1, self.loan2_c1] # At least 2 samples
        with patch('src.risk_models.pd_model.ModelRegistry'), patch('src.risk_models.pd_model.joblib.dump'):
            self.pd_model.train(self.mock_kb_service)

        self.assertIsNotNone(self.pd_model.model)
        self.assertTrue(hasattr(self.pd_model, '_feature_names') and self.pd_model._feature_names)

        mock_explainer_instance = MockTreeExplainer.return_value
        # SHAP values for binary classification: list of two arrays [class0_shap, class1_shap]
        # Number of features after preprocessing
        num_processed_features = len(self.pd_model._feature_names)
        mock_shap_values = [np.random.rand(1, num_processed_features), np.random.rand(1, num_processed_features)]
        mock_explainer_instance.shap_values.return_value = mock_shap_values

        # Prepare sample data for SHAP
        raw_features_df = self.pd_model._prepare_features(self.mock_kb_service)
        sample_instance_df = raw_features_df.drop(['default_status', 'loan_id', 'company_id'], axis=1).head(1)

        shap_importances = self.pd_model.get_feature_importance_shap(sample_instance_df)

        self.assertIsNotNone(shap_importances)
        self.assertIsInstance(shap_importances, dict)
        self.assertEqual(len(shap_importances), num_processed_features)
        self.assertTrue(all(f_name in shap_importances for f_name in self.pd_model._feature_names))
        MockTreeExplainer.assert_called_once_with(self.pd_model.model.named_steps['classifier'])

        # Test SHAP when model not trained
        self.pd_model.model = None
        self.assertIsNone(self.pd_model.get_feature_importance_shap(sample_instance_df))


if __name__ == '__main__':
    unittest.main()
