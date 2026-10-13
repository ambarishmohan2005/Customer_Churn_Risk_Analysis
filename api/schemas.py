from typing import Literal, List, Optional
from pydantic import BaseModel, Field, model_validator


class CustomerInput(BaseModel):
    """Schema representing a single customer's features for churn prediction."""

    gender: Literal['Male', 'Female'] = Field(
        default='Female', description="Customer's gender"
    )
    SeniorCitizen: Literal[0, 1] = Field(
        default=0, description="Whether the customer is a senior citizen (1) or not (0)"
    )
    Partner: Literal['Yes', 'No'] = Field(
        default='No', description="Whether the customer has a partner"
    )
    Dependents: Literal['Yes', 'No'] = Field(
        default='No', description="Whether the customer has dependents"
    )
    tenure: int = Field(
        default=1, ge=0, le=120, description="Number of months the customer has stayed with the company"
    )
    PhoneService: Literal['Yes', 'No'] = Field(
        default='Yes', description="Whether the customer has phone service"
    )
    MultipleLines: Literal['No phone service', 'No', 'Yes'] = Field(
        default='No', description="Whether the customer has multiple lines"
    )
    InternetService: Literal['DSL', 'Fiber optic', 'No'] = Field(
        default='Fiber optic', description="Customer’s internet service provider"
    )
    OnlineSecurity: Literal['No', 'Yes', 'No internet service'] = Field(
        default='No', description="Whether the customer has online security"
    )
    OnlineBackup: Literal['Yes', 'No', 'No internet service'] = Field(
        default='No', description="Whether the customer has online backup"
    )
    DeviceProtection: Literal['No', 'Yes', 'No internet service'] = Field(
        default='No', description="Whether the customer has device protection"
    )
    TechSupport: Literal['No', 'Yes', 'No internet service'] = Field(
        default='No', description="Whether the customer has tech support"
    )
    StreamingTV: Literal['No', 'Yes', 'No internet service'] = Field(
        default='No', description="Whether the customer has streaming TV"
    )
    StreamingMovies: Literal['No', 'Yes', 'No internet service'] = Field(
        default='No', description="Whether the customer has streaming movies"
    )
    Contract: Literal['Month-to-month', 'One year', 'Two year'] = Field(
        default='Month-to-month', description="The contract term of the customer"
    )
    PaperlessBilling: Literal['Yes', 'No'] = Field(
        default='Yes', description="Whether the customer has paperless billing"
    )
    PaymentMethod: Literal[
        'Electronic check',
        'Mailed check',
        'Bank transfer (automatic)',
        'Credit card (automatic)',
    ] = Field(default='Electronic check', description="Customer's payment method")
    MonthlyCharges: float = Field(
        default=70.0, ge=0.0, le=1000.0, description="The amount charged to the customer monthly"
    )
    TotalCharges: Optional[float] = Field(
        default=None, ge=0.0, description="The total amount charged to the customer (auto-calculated if omitted)"
    )

    @model_validator(mode='after')
    def compute_total_charges(self) -> 'CustomerInput':
        if self.TotalCharges is None:
            self.TotalCharges = round(float(self.tenure * self.MonthlyCharges), 2)
        return self

    model_config = {
        "json_schema_extra": {
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "No",
                "Dependents": "No",
                "tenure": 2,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 70.35,
                "TotalCharges": 140.70,
            }
        }
    }


class PredictionResponse(BaseModel):
    """Schema representing churn prediction result."""

    churn_prediction: int = Field(description="Binary churn prediction: 1 for Churn, 0 for Retain")
    churn_label: str = Field(description="'Yes' or 'No'")
    churn_probability: float = Field(description="Probability of churn between 0.0 and 1.0")
    risk_level: Literal['High', 'Medium', 'Low'] = Field(description="Assessed risk level")
    confidence: Literal['High', 'Medium', 'Low'] = Field(description="Confidence grade based on distance from 0.5")
    key_recommendation: str = Field(description="Actionable business retention advice")


class BatchCustomerInput(BaseModel):
    """Schema for batch customer prediction."""

    customers: List[CustomerInput] = Field(..., min_length=1, max_length=5000)


class BatchPredictionResponse(BaseModel):
    """Schema for batch prediction responses."""

    total_records: int
    churn_count: int
    non_churn_count: int
    overall_churn_rate: float
    predictions: List[PredictionResponse]


class HealthResponse(BaseModel):
    """Service health response schema."""

    status: str
    model_loaded: bool
    version: str
    num_features: int


class ModelMetadataResponse(BaseModel):
    """Model metadata response schema."""

    model_type: str
    features: List[str]
    target: str
    imbalance_strategy: str
    training_roc_auc: float
