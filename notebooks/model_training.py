# notebooks/model_training.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.model import create_cnn_bilstm_attention, compile_model, AttentionLayer
from src.utils import load_config
from src.embeddings import load_fasttext_embeddings

def create_callbacks(config):
    """Callback'leri oluştur - TensorBoard OLMADAN"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    callbacks = [
        # Early stopping
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Learning rate reduction
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        
        # Model checkpoint
        ModelCheckpoint(
            filepath=f"models/best_model_{timestamp}.h5",
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    return callbacks

def plot_training_history(history):
    """Eğitim geçmişini görselleştir"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[0, 0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Model Loss During Training')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # MAE
    axes[0, 1].plot(history.history['mae'], label='Train MAE', linewidth=2)
    axes[0, 1].plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].set_title('Mean Absolute Error During Training')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # RMSE
    axes[1, 0].plot(history.history['rmse'], label='Train RMSE', linewidth=2)
    axes[1, 0].plot(history.history['val_rmse'], label='Validation RMSE', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('RMSE')
    axes[1, 0].set_title('Root Mean Squared Error During Training')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Learning Rate
    if 'lr' in history.history:
        axes[1, 1].plot(history.history['lr'], label='Learning Rate', linewidth=2, color='purple')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_yscale('log')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        # Epoch sayısını göster
        epochs = range(1, len(history.history['loss']) + 1)
        axes[1, 1].plot(epochs, history.history['loss'], alpha=0.3, label='Train Loss', color='blue')
        axes[1, 1].plot(epochs, history.history['val_loss'], alpha=0.3, label='Val Loss', color='orange')
        axes[1, 1].fill_between(epochs, history.history['loss'], history.history['val_loss'], alpha=0.1)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_title('Loss Convergence')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('models/training_history.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig

def evaluate_model(model, X_test, y_test, X_val, y_val):
    """Modeli değerlendir"""
    print("\n📊 MODEL DEĞERLENDİRMESİ")
    print("=" * 60)
    
    # Tahminler
    y_pred_test = model.predict(X_test, verbose=0).flatten()
    y_pred_val = model.predict(X_val, verbose=0).flatten()
    
    # Metrikler
    test_metrics = {
        'MSE': mean_squared_error(y_test, y_pred_test),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'MAE': mean_absolute_error(y_test, y_pred_test),
        'R2': r2_score(y_test, y_pred_test)
    }
    
    val_metrics = {
        'MSE': mean_squared_error(y_val, y_pred_val),
        'RMSE': np.sqrt(mean_squared_error(y_val, y_pred_val)),
        'MAE': mean_absolute_error(y_val, y_pred_val),
        'R2': r2_score(y_val, y_pred_val)
    }
    
    # Metrikleri yazdır
    print("Test Seti Metrikleri:")
    print("-" * 40)
    for metric, value in test_metrics.items():
        print(f"  {metric:<6}: {value:>8.4f}")
    
    print("\nValidation Seti Metrikleri:")
    print("-" * 40)
    for metric, value in val_metrics.items():
        print(f"  {metric:<6}: {value:>8.4f}")
    
    # Örnek tahminler
    print("\n🎯 ÖRNEK TAHMİNLER (Test Setinden)")
    print("-" * 60)
    
    sample_indices = np.random.choice(len(y_test), min(10, len(y_test)), replace=False)
    
    print(f"{'Index':<8} {'Gerçek':<8} {'Tahmin':<8} {'Hata':<8} {'Durum':<15}")
    print("-" * 60)
    
    for idx in sample_indices:
        true_val = y_test[idx]
        pred_val = y_pred_test[idx]
        error = abs(true_val - pred_val)
        
        # Durum belirle
        if error <= 1.0:
            status = "✅ Çok İyi"
        elif error <= 2.0:
            status = "👍 İyi"
        elif error <= 3.0:
            status = "⚠️  Orta"
        else:
            status = "❌ Kötü"
        
        print(f"{idx:<8} {true_val:<8.2f} {pred_val:<8.2f} {error:<8.2f} {status:<15}")
    
    return test_metrics, val_metrics, y_pred_test

def plot_predictions(y_true, y_pred, title="Gerçek vs Tahmin Değerleri"):
    """Gerçek ve tahmin değerlerini görselleştir"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Scatter plot
    axes[0].scatter(y_true, y_pred, alpha=0.5, s=20)
    axes[0].plot([-10, 10], [-10, 10], 'r--', alpha=0.5, label='Mükemmel Tahmin')
    axes[0].set_xlabel('Gerçek Değerler')
    axes[0].set_ylabel('Tahmin Değerleri')
    axes[0].set_title(f'{title}\nScatter Plot')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([-11, 11])
    axes[0].set_ylim([-11, 11])
    
    # Hata dağılımı
    errors = y_true - y_pred
    axes[1].hist(errors, bins=30, edgecolor='black', alpha=0.7)
    axes[1].axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Sıfır Hata')
    axes[1].axvline(x=np.mean(errors), color='green', linestyle='-', alpha=0.7, 
                   label=f'Ortalama Hata: {np.mean(errors):.3f}')
    axes[1].set_xlabel('Hata (Gerçek - Tahmin)')
    axes[1].set_ylabel('Frekans')
    axes[1].set_title('Hata Dağılımı')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('models/predictions_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig

def main():
    print("🚀 CNN-BiLSTM with Attention Model Eğitimi")
    print("=" * 60)
    
    # 1. Config yükle
    print("1. Yapılandırma yükleniyor...")
    config = load_config()
    print(f"   ✅ Config yüklendi: {config['project']['name']} v{config['project']['version']}")
    
    # 2. İşlenmiş verileri yükle
    print("\n2. İşlenmiş veriler yükleniyor...")
    with open('data/processed/processed_data.pickle', 'rb') as f:
        processed_data = pickle.load(f)
    
    X_train = processed_data['X_train']
    X_val = processed_data['X_val']
    X_test = processed_data['X_test']
    y_train = processed_data['y_train']
    y_val = processed_data['y_val']
    y_test = processed_data['y_test']
    tokenizer = processed_data['tokenizer']
    vocab_size = processed_data['vocab_size']
    
    print(f"   ✅ Veriler yüklendi:")
    print(f"      Train:      {X_train.shape}")
    print(f"      Validation: {X_val.shape}")
    print(f"      Test:       {X_test.shape}")
    print(f"      Vocab size: {vocab_size}")
    
    # 3. FastText embedding matrix oluştur (opsiyonel)
    embedding_matrix = None
    if config['model']['use_fasttext']:
        print("\n3. FastText embedding'leri yükleniyor...")
        try:
            embedding_matrix = load_fasttext_embeddings(
                tokenizer, 
                embedding_dim=config['model']['embedding_dim']
            )
            print(f"   ✅ FastText embedding matrix oluşturuldu: {embedding_matrix.shape}")
        except Exception as e:
            print(f"   ⚠️  FastText yüklenemedi: {e}")
            print("   ℹ️  Random embedding kullanılacak")
            config['model']['use_fasttext'] = False
    else:
        print("\n3. Random embedding kullanılacak")
    
    # 4. Model oluştur
    print("\n4. Model oluşturuluyor...")
    import tensorflow as tf

    # Çalışan basit CNN-BiLSTM modeli
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=config['model']['embedding_dim'],
            input_length=config['data']['max_sequence_length'],
            mask_zero=True
        ),
        
        # CNN 1
        tf.keras.layers.Conv1D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Dropout(0.3),
        
        # CNN 2
        tf.keras.layers.Conv1D(32, 5, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Dropout(0.3),
        
        # BiLSTM
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(32, return_sequences=True, dropout=0.2)
        ),
        
        # Global pooling (Attention yerine)
        tf.keras.layers.GlobalAveragePooling1D(),
        
        # Dense katmanları
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        
        tf.keras.layers.Dense(16, activation='relu'),
        
        # Çıktı katmanı
        tf.keras.layers.Dense(1, activation='linear')
    ])  
        
    # Model summary
    model.summary()
    
    # 5. Modeli compile et
    print("\n5. Model compile ediliyor...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config['training']['learning_rate']),
        loss='mse',
        metrics=['mae', tf.keras.metrics.RootMeanSquaredError(name='rmse')]
    )
    
    print(f"   ✅ Model compile edildi")
    print(f"      Optimizer: Adam (lr={config['training']['learning_rate']})")
    print(f"      Loss: MSE")
    print(f"      Metrics: MAE, RMSE")
    
    # 6. Callback'leri oluştur
    print("\n6. Callback'ler oluşturuluyor...")
    callbacks = create_callbacks(config)
    print(f"   ✅ {len(callbacks)} callback oluşturuldu")
    
    # 7. Modeli eğit
    print("\n7. Model eğitimi başlıyor...")
    print("=" * 60)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config['training']['epochs'],
        batch_size=config['training']['batch_size'],
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n✅ Model eğitimi tamamlandı!")
    
    # 8. Eğitim geçmişini görselleştir
    print("\n8. Eğitim geçmişi görselleştiriliyor...")
    plot_training_history(history)
    
    # 9. Modeli değerlendir
    print("\n9. Model değerlendiriliyor...")
    test_metrics, val_metrics, y_pred_test = evaluate_model(model, X_test, y_test, X_val, y_val)
    
    # 10. Tahminleri görselleştir
    print("\n10. Tahminler görselleştiriliyor...")
    plot_predictions(y_test, y_pred_test)
    
    # 11. Modeli kaydet
    print("\n11. Model kaydediliyor...")
    model.save(config['paths']['trained_model'])
    
    # Eğitim geçmişini kaydet
    with open('models/training_history.pickle', 'wb') as f:
        pickle.dump(history.history, f)
    
    print(f"   ✅ Model kaydedildi: {config['paths']['trained_model']}")
    print(f"   ✅ Eğitim geçmişi kaydedildi: models/training_history.pickle")
    
    # 12. Final rapor
    print("\n" + "=" * 60)
    print("🎉 MODEL EĞİTİMİ BAŞARIYLA TAMAMLANDI!")
    print("=" * 60)
    
    print("\n📈 PERFORMANS ÖZETİ:")
    print("-" * 40)
    print(f"   Test RMSE:    {test_metrics['RMSE']:.4f}")
    print(f"   Test MAE:     {test_metrics['MAE']:.4f}")
    print(f"   Test R²:      {test_metrics['R2']:.4f}")
    print(f"   Val RMSE:     {val_metrics['RMSE']:.4f}")
    print(f"   Val MAE:      {val_metrics['MAE']:.4f}")
    print(f"   Val R²:       {val_metrics['R2']:.4f}")
    
    print("\n🔧 MODEL DETAYLARI:")
    print("-" * 40)
    print(f"   Model:        {config['model']['name']}")
    print(f"   Embedding:    {'FastText' if config['model']['use_fasttext'] else 'Random'}")
    print(f"   Epochs:       {len(history.history['loss'])}/{config['training']['epochs']}")
    print(f"   Batch size:   {config['training']['batch_size']}")
    print(f"   Vocab size:   {vocab_size}")
    
    print("\n📁 OLUŞTURULAN DOSYALAR:")
    print("-" * 40)
    print(f"   1. {config['paths']['trained_model']}")
    print(f"   2. models/training_history.pickle")
    print(f"   3. models/training_history.png")
    print(f"   4. models/predictions_analysis.png")
    print(f"   5. logs/ (TensorBoard logları)")
    
    print("\n🚀 BİR SONRAKİ ADIM:")
    print("-" * 40)
    print("   Demo uygulamasını çalıştırmak için:")
    print("   streamlit run app.py")

if __name__ == "__main__":
    # GPU kullanılabilirliğini kontrol et
    print("🔍 GPU Kontrolü...")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"   ✅ GPU bulundu: {len(gpus)} adet")
        for gpu in gpus:
            print(f"      {gpu}")
        # GPU memory büyümesine izin ver
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("   ⚠️  GPU bulunamadı, CPU kullanılacak")
    
    main()