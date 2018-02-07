"""
Definition of models.
"""

from django.db import models
#from django.contrib.gis.db import models

# Create your models here.

class Post(models.Model):
    text = models.TextField()
    author = models.CharField(max_length=100)

    def __str__(self):
        """A string representation of the model."""
        return self.text[:50]


