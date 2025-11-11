import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

def find_top_zones(img_gray_original, top_n=5, min_area_ratio=0.3, max_area_ratio=0.95):
    """
    Находим top_n зон на исходном изображении с адаптивным увеличением контраста.
    
    Args:
        img_gray_original: исходное изображение в градациях серого
        top_n: требуемое количество зон (будет уменьшаться до 1, если не найдено)
        min_area_ratio: минимальное соотношение площади зоны к самой большой (0.3 = 30%)
        max_area_ratio: максимальное соотношение площади зоны к изображению (0.95 = 95%)
    
    Returns:
        список контуров отсортированных слева направо
    """
    img_for_search = img_gray_original.copy()
    img_area = img_gray_original.shape[0] * img_gray_original.shape[1]

    # Нормализация изображения (аналог уровня с битовой глубиной)
    in_min, in_max = np.percentile(img_for_search, 1), np.percentile(img_for_search, 99)
    img_for_search = (img_for_search - in_min) / (in_max - in_min) * 255
    img_for_search = np.clip(img_for_search, 0, 255).astype(np.uint8)

    # Параметры для адаптивного поиска
    clipLimit_values = [2.0, 3.0, 4.0, 5.0, 7.0, 10.0]  # Постепенно увеличиваем контраст
    threshold_values = [30, 25, 20, 15, 10]  # Уменьшаем порог
    
    best_contours = []
    current_target = top_n  # Начинаем с желаемого количества зон
    
    # Пробуем найти от top_n до 1 зоны
    while current_target >= 1 and len(best_contours) < current_target:
        for clipLimit in clipLimit_values:
            for threshold_val in threshold_values:
                # Применяем CLAHE с текущими параметрами
                clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(8, 8))
                img_clahe = clahe.apply(img_for_search)
                
                # Пробуем найти контуры
                _, mask = cv2.threshold(img_clahe, threshold_val, 255, cv2.THRESH_BINARY)
                kernel = np.ones((3, 3), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if len(contours) == 0:
                    continue
                
                # Сортируем по площади
                contours = sorted(contours, key=cv2.contourArea, reverse=True)
                
                # Фильтруем контуры: исключаем слишком большие и маленькие
                good_contours = []
                for cnt in contours[:current_target * 3]:  # Проверяем больше контуров
                    area = cv2.contourArea(cnt)
                    area_ratio_to_image = area / img_area
                    
                    # ВАЖНО: Исключаем зоны размером почти с все изображение
                    if area_ratio_to_image > max_area_ratio:
                        continue
                    
                    # Проверяем минимальный размер относительно самой большой допустимой зоны
                    if len(good_contours) > 0:
                        max_good_area = cv2.contourArea(good_contours[0])
                        if area < max_good_area * min_area_ratio:
                            continue
                    
                    good_contours.append(cnt)
                
                # Если нашли достаточно хороших зон для текущей цели
                if len(good_contours) >= current_target:
                    best_contours = good_contours[:current_target]
                    break
                
                # Сохраняем лучший результат
                if len(good_contours) > len(best_contours):
                    best_contours = good_contours[:current_target]
            
            # Если нашли достаточно зон, выходим
            if len(best_contours) >= current_target:
                break
        
        # Если нашли нужное количество, проверяем качество
        if len(best_contours) >= current_target:
            # Проверяем, что все зоны соответствуют критериям
            all_good = True
            if len(best_contours) > 0:
                max_area = cv2.contourArea(best_contours[0])
                for cnt in best_contours:
                    area = cv2.contourArea(cnt)
                    if area < max_area * min_area_ratio:
                        all_good = False
                        break
            
            if all_good:
                break
        
        # Если не нашли current_target зон, пробуем найти меньше
        current_target -= 1
    
    # Если ничего не нашли, используем базовый алгоритм
    if len(best_contours) == 0:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img_for_search)
        _, mask = cv2.threshold(img_clahe, 30, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Фильтруем только по размеру изображения
        filtered = [c for c in contours if cv2.contourArea(c) / img_area <= max_area_ratio]
        best_contours = sorted(filtered, key=cv2.contourArea, reverse=True)[:top_n]
    
    # Сортировка контуров слева направо
    best_contours = sorted(best_contours, key=lambda c: cv2.boundingRect(c)[0])
    return best_contours

def contrast_zones_black_bg(img_gray_original, contours, show_zones=True):
    """Вырезаем зоны из исходника, оставляем черный фон."""
    result = np.zeros((img_gray_original.shape[0], img_gray_original.shape[1], 3), dtype=np.uint16)

    # Вычисляем площади для отображения
    if contours and len(contours) > 0:
        max_area = cv2.contourArea(contours[0]) if contours else 0
    
    for idx, cnt in enumerate(contours, 1):
        x, y, w, h = cv2.boundingRect(cnt)
        roi_original = img_gray_original[y:y+h, x:x+w].copy()
        if roi_original.size < 100:
            continue

        result[y:y+h, x:x+w, :] = cv2.merge([roi_original]*3)

        if show_zones:
            area = cv2.contourArea(cnt)
            area_ratio = (area / max_area * 100) if max_area > 0 else 0
            
            # Цвет рамки в зависимости от размера зоны
            if area_ratio >= 30:
                color = (0, 255, 0)  # Зеленый - хорошая зона
            elif area_ratio >= 15:
                color = (0, 255, 255)  # Желтый - средняя зона
            else:
                color = (0, 165, 255)  # Оранжевый - маленькая зона
            
            cv2.rectangle(result, (x, y), (x+w, y+h), color, 2)
            
            # Отображаем номер и процент от максимальной зоны
            label = f"{idx} ({area_ratio:.0f}%)"
            cv2.putText(result, label, (x+5, y+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return result

def process_single_image(img_path, output_dir, top_n=5):
    """Обрабатывает одно изображение и сохраняет вырезанные зоны."""
    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return 0, f"Ошибка чтения: {img_path.name}"

    if img.dtype == np.uint16:
        img = (img / 4).astype(np.uint16)  # Преобразуем в 10 бит (диапазон 0-1023)

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    contours = find_top_zones(img, top_n=top_n)
    
    if not contours:
        return 0, f"Зоны не найдены: {img_path.name}"

    # Создаем подпапку для каждого изображения
    img_output_dir = output_dir / img_path.stem
    img_output_dir.mkdir(parents=True, exist_ok=True)

    zones_saved = 0
    for idx, cnt in enumerate(contours, 1):
        x, y, w, h = cv2.boundingRect(cnt)
        roi = img[y:y+h, x:x+w]

        if roi.dtype != np.uint16:
            roi = np.uint16(roi)

        zone_path = img_output_dir / f"zone_{idx}.png"
        cv2.imwrite(str(zone_path), roi)
        zones_saved += 1

    return zones_saved, f"Обработано: {img_path.name}"

# === GUI ===
class ContrastApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Зональное контрастирование с черным фоном")
        self.img_original = None
        self.result_img = None
        self.processed_result = None
        self.contours = None

        control_frame = tk.Frame(root)
        control_frame.pack(pady=10)
        
        tk.Button(control_frame, text="Загрузить изображение", 
                 command=self.load_image, width=20).pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Сохранить результат", 
                 command=self.save_image, width=20).pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Вырезать зоны", 
                 command=self.cut_and_save_zones, width=20).pack(side=tk.LEFT, padx=5)
        
        # Новая кнопка для пакетной обработки
        batch_frame = tk.Frame(root)
        batch_frame.pack(pady=5)
        tk.Button(batch_frame, text="📁 Обработать папку", 
                 command=self.batch_process_folder, 
                 width=30, height=2, 
                 bg='lightblue', font=('Arial', 10, 'bold')).pack()
        
        self.show_zones = tk.BooleanVar(value=True)
        tk.Checkbutton(control_frame, text="Показать зоны", 
                      variable=self.show_zones, 
                      command=self.update_display).pack(side=tk.LEFT, padx=5)

        self.canvas = tk.Canvas(root, width=1024, height=1024, bg='gray')
        self.canvas.pack(pady=5)

        self.status = tk.Label(root, text="Загрузите изображение для начала работы", 
                              bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

    def batch_process_folder(self):
        """Пакетная обработка всех изображений в папке с многопоточностью."""
        input_dir = filedialog.askdirectory(title="Выберите папку с изображениями")
        if not input_dir:
            return

        output_dir = filedialog.askdirectory(title="Выберите папку для сохранения результатов")
        if not output_dir:
            return

        input_path = Path(input_dir)
        output_path = Path(output_dir)

        # Поддерживаемые форматы
        supported_formats = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']
        image_files = []
        for ext in supported_formats:
            image_files.extend(input_path.glob(f'*{ext}'))
            image_files.extend(input_path.glob(f'*{ext.upper()}'))
        
        # Убираем дубликаты (если файл найден и в нижнем, и в верхнем регистре)
        image_files = list(set(image_files))

        if not image_files:
            messagebox.showwarning("Предупреждение", 
                                  "В выбранной папке не найдено изображений")
            return

        # Определяем количество потоков (по умолчанию = количество ядер CPU)
        num_threads = os.cpu_count() or 4

        # Окно прогресса
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Обработка изображений")
        progress_window.geometry("600x200")
        
        progress_label = tk.Label(progress_window, 
                                 text=f"Найдено изображений: {len(image_files)}\nИспользуется потоков: {num_threads}\nНачинаем обработку...",
                                 font=('Arial', 10))
        progress_label.pack(pady=10)
        
        # Прогресс-бар
        progress_bar_frame = tk.Frame(progress_window)
        progress_bar_frame.pack(pady=5, padx=20, fill=tk.X)
        
        progress_canvas = tk.Canvas(progress_bar_frame, width=550, height=30, bg='white', highlightthickness=1)
        progress_canvas.pack()
        
        progress_text = tk.Text(progress_window, height=6, width=70)
        progress_text.pack(pady=10, padx=10)
        
        progress_window.update()

        total_zones = 0
        processed_count = 0
        lock = threading.Lock()

        def update_progress():
            """Обновление прогресс-бара."""
            with lock:
                percent = (processed_count / len(image_files)) * 100
                bar_width = int((processed_count / len(image_files)) * 550)
                
                progress_canvas.delete("all")
                progress_canvas.create_rectangle(0, 0, bar_width, 30, fill='#4CAF50', outline='')
                progress_canvas.create_text(275, 15, text=f"{processed_count}/{len(image_files)} ({percent:.1f}%)", 
                                          font=('Arial', 10, 'bold'))
                
                progress_label.config(text=f"Обработано: {processed_count}/{len(image_files)} ({percent:.1f}%)\n"
                                          f"Вырезано зон: {total_zones}\n"
                                          f"Потоков: {num_threads}")
                progress_window.update()

        # Многопоточная обработка
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Отправляем все задачи в пул
            futures = {executor.submit(process_single_image, img_file, output_path): img_file 
                      for img_file in image_files}
            
            # Обрабатываем результаты по мере завершения
            for future in as_completed(futures):
                img_file = futures[future]
                try:
                    zones_count, message = future.result()
                    
                    with lock:
                        total_zones += zones_count
                        processed_count += 1
                    
                    progress_text.insert(tk.END, f"{message} ({zones_count} зон)\n")
                    progress_text.see(tk.END)
                    update_progress()
                    
                except Exception as e:
                    with lock:
                        processed_count += 1
                    progress_text.insert(tk.END, f"Ошибка: {img_file.name} - {str(e)}\n")
                    progress_text.see(tk.END)
                    update_progress()

        # Итоговое сообщение
        final_message = f"\n✓ Обработка завершена!\n"
        final_message += f"Обработано изображений: {processed_count}\n"
        final_message += f"Всего вырезано зон: {total_zones}\n"
        final_message += f"Результаты сохранены в: {output_path}"
        
        progress_text.insert(tk.END, final_message)
        progress_text.see(tk.END)
        
        # Заполняем прогресс-бар на 100%
        progress_canvas.delete("all")
        progress_canvas.create_rectangle(0, 0, 550, 30, fill='#4CAF50', outline='')
        progress_canvas.create_text(275, 15, text="100% - Готово!", font=('Arial', 10, 'bold'))
        
        messagebox.showinfo("Готово", final_message)
        self.status.config(text=f"Пакетная обработка завершена: {processed_count} изображений, {total_zones} зон")

    def load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp")])
        if path:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                self.status.config(text="Ошибка загрузки изображения")
                return

            if img.dtype == np.uint16:
                img = (img / 4).astype(np.uint16)  # Преобразуем в 10 бит (диапазон 0-1023)

            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            self.img_original = img
            self.status.config(text=f"Загружено: {self.img_original.shape[1]}x{self.img_original.shape[0]}")
            self.update_image()

    def save_image(self):
        if self.processed_result is None or self.img_original is None:
            self.status.config(text="Нет изображения для сохранения")
            return
            
        path = filedialog.asksaveasfilename(
            defaultextension=".tiff",
            filetypes=[("TIFF", "*.tiff"), ("PNG", "*.png"), ("JPEG", "*.jpg"), ("All files", "*.*")])
        if path:
            # Преобразуем изображение в 10-битное перед сохранением (диапазон 0-1023)
            if self.processed_result.dtype != np.uint16:
                self.processed_result = np.uint16(self.processed_result)  # Преобразуем в 10 бит
            cv2.imwrite(path, self.processed_result)
            self.status.config(text=f"Сохранено: {path}")

    def cut_and_save_zones(self):
        if self.contours is None or self.img_original is None:
            self.status.config(text="Нет зон для вырезания")
            return

        save_dir = filedialog.askdirectory(title="Выберите папку для сохранения зон")
        if not save_dir:
            return

        for idx, cnt in enumerate(self.contours, 1):
            x, y, w, h = cv2.boundingRect(cnt)
            roi = self.img_original[y:y+h, x:x+w]

            # Преобразуем изображение в 10-битное перед сохранением (диапазон 0-1023)
            if roi.dtype != np.uint16:
                roi = np.uint16(roi)

            zone_path = os.path.join(save_dir, f"zone_{idx}.png")
            cv2.imwrite(zone_path, roi)
        
        self.status.config(text=f"Зоны сохранены в: {save_dir}")

    def update_image(self):
        if self.img_original is None:
            return

        self.status.config(text="Обработка...")
        self.root.update()

        self.contours = find_top_zones(self.img_original, top_n=5)
        self.processed_result = contrast_zones_black_bg(
            self.img_original, self.contours, show_zones=self.show_zones.get())
        
        self.update_display()
        
        # Добавляем информацию о качестве поиска
        zones_count = len(self.contours)
        if zones_count == 5:
            status_msg = f"✓ Обработано | Найдено зон: {zones_count} (отлично!)"
        elif zones_count >= 3:
            status_msg = f"⚠ Обработано | Найдено зон: {zones_count} из 5 (хорошо)"
        else:
            status_msg = f"⚠ Обработано | Найдено зон: {zones_count} из 5 (проверьте изображение)"
        
        self.status.config(text=status_msg)

    def update_display(self):
        if self.processed_result is None or self.img_original is None:
            return

        # Преобразуем изображение в 8-бит для отображения
        img_display = np.uint8(self.processed_result // 4)  # Преобразуем 10-бит в 8-бит

        img_pil = Image.fromarray(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB))
        
        canvas_size = 1024
        img_w, img_h = img_pil.size
        scale = min(canvas_size / img_w, canvas_size / img_h)
        new_w, new_h = int(img_w * scale), int(img_h * scale)
        img_pil = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)

        self.result_img = ImageTk.PhotoImage(img_pil)
        self.canvas.delete("all")
        self.canvas.create_image(canvas_size//2, canvas_size//2, 
                                anchor=tk.CENTER, image=self.result_img)

if __name__ == "__main__":
    root = tk.Tk()
    app = ContrastApp(root)
    root.mainloop()