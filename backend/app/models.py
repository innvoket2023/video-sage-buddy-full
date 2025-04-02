from app.extensions import db
import uuid
from sqlalchemy.dialects.postgresql import UUID
import datetime

class User(db.Model):
    __tablename__ = 'users'
    
    user_id = db.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = db.Column(db.String(100), unique=True, nullable=False)
    email = db.Column(db.String(255), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime(timezone=True), server_default=db.func.now())
    updated_at = db.Column(db.DateTime(timezone=True), server_default=db.func.now(), onupdate=db.func.now())
    last_login = db.Column(db.DateTime(timezone=True), nullable=True, onupdate=db.func.now())
    is_active = db.Column(db.Boolean, default=False)

    videos = db.relationship("Video", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f'<User {self.username}>'
    
    #Helper function
    def to_dict(self):
        return {
            'id': self.user_id,
            'username': self.username,
            'email': self.email,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'last_login': self.last_login,
            'is_active': self.is_active 
        }

class Video(db.Model):
    __tablename__ = 'videos'
    
    video_id = db.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = db.Column(UUID(as_uuid=True), db.ForeignKey('users.user_id', ondelete='CASCADE'), nullable=False)
    safe_name_for_vectordb = db.Column(db.String(255), nullable=True)
    title = db.Column(db.String(255), nullable=False)
    transcript = db.Column(db.Text, nullable=True)
    description = db.Column(db.Text, nullable=True)
    cloudinary_public_id = db.Column(db.String(255), nullable=False)
    cloudinary_url = db.Column(db.Text, nullable=False)
    cloudinary_resource_type = db.Column(db.String(50), nullable=True)
    duration = db.Column(db.Numeric, nullable=True)
    upload_date = db.Column(db.DateTime(timezone=True), server_default=db.func.now())
    status = db.Column(db.String(50), default='active')
    voice_id = db.Column(db.String(255), nullable=True)
    
    # Add metadata fields similar to User
    updated_at = db.Column(db.DateTime(timezone=True), server_default=db.func.now(), onupdate=db.func.now())
    view_count = db.Column(db.Integer, default=0)
    
    # Define relationship back to user
    user = db.relationship("User", back_populates="videos")
    
    # Add validation for status field
    __table_args__ = (
        db.CheckConstraint(status.in_(['processing', 'active', 'archived']), name='status_check'),
    )
    
    def __repr__(self):
        return f'<Video {self.title} ({self.video_id})>'
    
    # Helper functions
    def to_dict(self):
        return {
            'id': str(self.video_id),
            'user_id': str(self.user_id),
            'title': self.title,
            'description': self.description,
            'cloudinary_public_id': self.cloudinary_public_id,
            'cloudinary_url': self.cloudinary_url,
            'cloudinary_resource_type': self.cloudinary_resource_type,
            'duration': float(self.duration) if self.duration else None,
            'upload_date': self.upload_date.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'status': self.status,
            'view_count': self.view_count,
            'username': self.user.username if self.user else None
        }
    
    def increment_view(self):
        """Increment the view count for this video"""
        self.view_count += 1
        return self.view_count
    
    @property
    def is_active(self):
        """Check if the video is in active status"""
        return self.status == 'active'
    
    @property
    def formatted_duration(self):
        """Return the duration in a human-readable format (minutes:seconds)"""
        if not self.duration:
            return "0:00"
        
        total_seconds = int(float(self.duration))
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        return f"{minutes}:{seconds:02d}"
