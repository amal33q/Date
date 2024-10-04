from flask import Flask, render_template, request
import cv2
import numpy as np
import tensorflow as tf
import base64


app = Flask(__name__)

# تحميل نموذج TensorFlow المتدرب مسبقًا
model = tf.keras.models.load_model('date_quality_model1.h5')

# دالة تصنيف التمور
def classify_date(image):
    resized_image = cv2.resize(image, (150, 150))  # على  أن النموذج يتوقع
    resized_image = resized_image / 255.0  # تطبيع البيانات
    resized_image = np.expand_dims(resized_image, axis=0)  # إضافة بعد إضافي

    # تمرير الصورة إلى النموذج للحصول على التنبؤ
    prediction = model.predict(resized_image)
    print("Prediction:", prediction)  # طباعة النتيجة

    # تحديد النتيجة بناءً على التنبؤ
    if prediction[0][0] > 0.7:
        return "جودة عالية"
    elif prediction[0][1] > 0.7:
        return "جودة متوسطة"
    else:
        return "غير صالحة"


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/load_image', methods=['POST'])
def load_image():
    # استخدام رفع الملفات
    if 'image' in request.files:
        file = request.files['image']
        if file.filename == '':
            return "يجب اختيار صورة"

        # قراءة الصورة باستخدام OpenCV
        file.save("uploaded_image.jpg")  # حفظ الصورة بشكل مؤقت
        image = cv2.imread("uploaded_image.jpg")

    # استخدام Base64
    elif 'image' in request.form:
        image_data = request.form['image']
        header, encoded = image_data.split(',', 1)
        decoded = base64.b64decode(encoded)

        # حفظ الصورة في ملف مؤقت
        with open("uploaded_image.jpg", "wb") as f:
            f.write(decoded)

        # قراءة الصورة باستخدام OpenCV
        image = cv2.imread("uploaded_image.jpg")

    else:
        return "لا توجد صورة لتحميلها"

    # تحليل الصورة وتصنيف التمور
    result = classify_date(image)

    return render_template('result.html', result=result)


def capture_image():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Camera not accessible.")
        return None

    ret, frame = cap.read()

    if ret:
        # تحسين الصورة
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # تحويل للصورة الرمادية
        enhanced_image = cv2.equalizeHist(gray_image)  # تحسين التباين
        enhanced_image = cv2.GaussianBlur(enhanced_image, (5, 5), 0)  # إزالة الضوضاء

        # عرض الصورة المحسنة
        cv2.imshow("Captured Image", enhanced_image)
        cv2.waitKey(0)

        # تحليل الصورة وتصنيف التمور
        result = classify_date(enhanced_image)
    else:
        print("Error: Could not read frame.")
        result = None

    cap.release()
    cv2.destroyAllWindows()

    return result


# في مكان آخر في الكود (مثل دالة عرض صفحة النتائج)
@app.route('/capture', methods=['GET', 'POST'])
def capture():
    result = capture_image()  # استدعاء دالة التقاط الصورة
    return render_template('result.html', result=result)  # إرجاع القالب مع النتيجة



if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
