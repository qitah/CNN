from django.db import models


class image(models.Model):
    image = models.ImageField(upload_to="images")
    lable = models.CharField(max_length=50, blank=True)

    def __str__(self):
        return self.lable
        
# Create your models here.
