import os
import numpy as np
import tensorflow as tf
import cv2
from django.utils import timezone
from django.conf import settings
from django.shortcuts import render
from .forms import ImageUploadForm
from django.core.files.storage import default_storage
from .gradcam_utils import generate_gradcam
from django.template.loader import get_template
from django.http import HttpResponse
from xhtml2pdf import pisa

# Load the best model once
MODEL_PATH = os.path.join('models', 'vgg16_optsgd_lr0.0001_bs8_ep30.keras')
model = tf.keras.models.load_model(MODEL_PATH)

CLASS_NAMES = ['Grade 0', 'Grade 1', 'Grade 2', 'Grade 3', 'Grade 4']
INTERPRETATIONS = [
    'Healthy or normal; no radiographic signs of OA.',
    'Doubtful joint space narrowing and possible osteophyte lipping.',
    'Definite osteophytes and possible joint space narrowing.',
    'Multiple osteophytes, definite joint space narrowing, possible bony deformity.',
    'Large osteophytes, marked joint space narrowing, severe sclerosis, and definite bony deformity.'
]

def index(request):
    form = ImageUploadForm()
    return render(request, 'index.html', {'form': form})

def results(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.cleaned_data['xray']
            path = default_storage.save('uploads/' + image.name, image)
            full_path = os.path.join('media', path)

            # Apply CLAHE and preprocess
            clahe_img = apply_clahe(full_path)
            clahe_path = os.path.join('media', 'uploads', 'clahe_' + os.path.basename(full_path))
            cv2.imwrite(clahe_path, clahe_img)

            img_array = preprocess_image(clahe_img)
            prediction = model.predict(np.expand_dims(img_array, axis=0))
            predicted_class = np.argmax(prediction)

            gradcam_path = generate_gradcam(model, clahe_path, predicted_class, layer_name="block5_conv3")

            request.session['pdf_data'] = {
                'image_path': os.path.join(settings.BASE_DIR, full_path),
                'gradcam_path': os.path.join(settings.BASE_DIR, gradcam_path),
                'predicted_class': CLASS_NAMES[predicted_class],
                'interpretation': INTERPRETATIONS[predicted_class],
                'filename': os.path.basename(full_path),
                'session_time': timezone.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            context = {
                'image_path': '/' + full_path,
                'gradcam_path': '/' + gradcam_path,
                'predicted_class': CLASS_NAMES[predicted_class],
                'interpretation': INTERPRETATIONS[predicted_class]
            }
            return render(request, 'results.html', context)

    return render(request, 'results.html', {'error': 'Invalid request'})

def preprocess_image(image):
    image = cv2.resize(image, (224, 224))
    img_array = image.astype("float32") / 255.0
    return img_array

def apply_clahe(image_path):
    image = cv2.imread(image_path)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final

def download_pdf(request):
    template_path = 'pdf_template.html'
    data = request.session.get('pdf_data', {})
    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = 'attachment; filename="result.pdf"'
    template = get_template(template_path)
    html = template.render(data)
    pisa_status = pisa.CreatePDF(html, dest=response)
    return response if not pisa_status.err else HttpResponse('Error generating PDF')
