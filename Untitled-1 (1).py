import tkinter as tk
from tkinter import messagebox, filedialog
from tkinter import font as tkfont
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(9, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x


# Функция для отображения данных
    root = tk.Tk()
    root.title("WhereWeather: Анализ климатических данных")
    root.geometry('500x450')
    frame1 = tk.Frame(root)
    frame1.pack()
    
    def display_data(data):
        headers = ['День', 'Время', 'Локация', 'Температура', 'Влажность', 'Скорость ветра', 'Направление ветра', 'Давление', 'Облачность']
        for col, header in enumerate(headers):
            header_label = tk.Label(data, text=header, padx=10, pady=10)
            header_label.grid(row=0, column=col)
    
        for row, data_row in data.iterrows():
            for col, value in enumerate(data_row):
                cell = tk.Label(data, text=value, padx=10, pady=10)
                cell.grid(row=row+1, column=col)

# Создание базы данных и таблицы пользователей
conn = sqlite3.connect('database.db')
cursor = conn.cursor()
cursor.execute('CREATE TABLE IF NOT EXISTS users (username TEXT, password TEXT)')
conn.commit()

def login_window():
    def login():
        username = entry_username.get()
        password = entry_password.get()
        
        cursor.execute('SELECT * FROM users WHERE username=? AND password=?', (username, password))
        user = cursor.fetchone()
         
        if user:
            messagebox.showinfo('Успешная авторизация', 'Авторизация прошла успешно!')
        else:
            messagebox.showerror('Ошибка авторизации', 'Неверное имя пользователя или пароль!')

    def register():
        username = username_entry.get()
        password = password_entry.get()

        cursor.execute('SELECT * FROM users WHERE username=?', (username,))
        row = cursor.fetchone()

        if row:
            messagebox.showinfo('Ошибка регистрации', 'Пользователь уже существует')
        else:
            cursor.execute('INSERT INTO users VALUES (?, ?)', (username, password))
            conn.commit()
            messagebox.showinfo('Успешная регистрация', 'Регистрация прошла успешно!')
            entry_username.delete(0, tk.END)
            entry_password.delete(0, tk.END)

    window = tk.Tk()
    window.title('Вход и регистрация')
    
    login_frame = tk.Frame(window)
    login_frame.pack(pady=10)
    
    label_username = tk.Label(login_frame, text='Логин:')
    label_username.pack()
    entry_username = tk.Entry(login_frame)
    entry_username.pack()
    
    label_password = tk.Label(login_frame, text='Пароль:')
    label_password.pack()
    entry_password = tk.Entry(login_frame, show='*')
    entry_password.pack()
    
    button_login = tk.Button(login_frame, text='Войти', command=login)
    button_login.pack()
    
    register_frame = tk.Frame(window)
    register_frame.pack(pady=10)
    
    username_label = tk.Label(register_frame, text="Имя пользователя:")
    username_label.pack()
    username_entry = tk.Entry(register_frame)
    username_entry.pack()

    password_label = tk.Label(register_frame, text="Пароль:")
    password_label.pack()
    password_entry = tk.Entry(register_frame, show="*")
    password_entry.pack()

    register_button = tk.Button(register_frame, text="Зарегистрироваться", command=register)
    register_button.pack()

    window.mainloop()

def main_menu():
    window = tk.Tk()

    def monitor_data():
        messagebox.showinfo('Мониторинг данных', 'Мониторинг данных...')

    def update_monitoring():
        start_date = start_date_entry.get()
        end_date = end_date_entry.get()
        
        monitoring_window = tk.Toplevel()
        monitoring_window.title("Мониторинг")

        map (location=[51.5074, -0.1278], zoom_start=10)

        start_date_label = tk.Label(monitoring_window, text="Начальная дата:")
        start_date_label.pack()
        start_date_entry = tk.Entry(monitoring_window)
        start_date_entry.insert(tk.END, "YYYY-MM-DD")
        start_date_entry.pack()

        end_date_label = tk.Label(monitoring_window, text="Конечная дата:")
        end_date_label.pack()
        end_date_entry = tk.Entry(monitoring_window)
        end_date_entry.insert(tk.END, "YYYY-MM-DD")
        end_date_entry.pack()

        update_button = tk.Button(monitoring_window, text="Обновить", command=update_monitoring)
        update_button.pack()

        map.save('monitoring_map.html')

        monitor_map = tk.Frame(monitoring_window, width=600, height=400)
        monitor_map.pack()
        monitor_map_html = tk.WebView.Create(monitor_map, width=600, height=400)
        monitor_map_html.load('monitoring_map.html')

    def open_files():
        root = main_menu.Tk()
        root.title("Анализ климатических данных")
        root.configure(bg="light blue")

        frame = tk.Frame(root, bg="light blue")
        frame.pack()

        choose_file_button = tk.Button(root, text="Выбрать файл", command=open_files, padx=10, pady=10)
        choose_file_button.pack()

        window.mainloop()

        messagebox.showinfo('Работа с файлами', 'Открытие файлов...')
        file_path = filedialog.askopenfilename()
        if file_path.endswith('.csv'):
            data = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            data = pd.read_excel(file_path)
        else:
            return
        data.destroy(data)

def analyze_data():
    messagebox.showinfo('Анализ данных', 'Анализ данных...')
    def analyze_data(self):
        if self.data is None:
            return

    location_window = tk.Toplevel()
    location_label = tk.Label(location_window, text="Введите локацию:")
    location_label.pack()
    location_entry = tk.Entry(location_window)
    location_entry.pack()

    start_time_window = tk.Toplevel()
    start_time_label = tk.Label(start_time_window, text="Введите начальное время (в формате ГГГГ-ММ-ДД ЧЧ:ММ:СС):")
    start_time_label.pack()
    start_time_entry = tk.Entry(start_time_window)
    start_time_entry.pack()

    end_time_window = tk.Toplevel()
    end_time_label = tk.Label(end_time_window, text="Введите конечное время (в формате ГГГГ-ММ-ДД ЧЧ:ММ:СС):")
    end_time_label.pack()
    end_time_entry = tk.Entry(end_time_window)
    end_time_entry.pack()
    
    submit_button = tk.Button(end_time_window, text="Submit", command=lambda: self.generate_subset(location_entry.get(), start_time_entry.get(), end_time_entry.get()))
    submit_button.pack()

def generate_subset(self, location, start_time, end_time, controller):
    subset = self.data[(self.data["локация"] == location) & (self.data["время"] >= start_time) & (self.data["время"] <= end_time)]

    subset.plot(x="день", y="температура", kind="line")
    plt.show()
def analyze_data(self):
    if self.data is None:
        return

    location_window = tk.Toplevel()
    location_label = tk.Label(location_window, text="Введите локацию:")
    location_label.pack()
    location_entry = tk.Entry(location_window)
    location_entry.pack()

    start_time_window = tk.Toplevel()
    start_time_label = tk.Label(start_time_window, text="Введите начальное время (в формате ГГГГ-ММ-ДД ЧЧ:ММ:СС):")
    start_time_label.pack()
    start_time_entry = tk.Entry(start_time_window)
    start_time_entry.pack()

    end_time_window = tk.Toplevel()
    end_time_label = tk.Label(end_time_window, text="Введите конечное время (в формате ГГГГ-ММ-ДД ЧЧ:ММ:СС):")
    end_time_label.pack()
    end_time_entry = tk.Entry(end_time_window)
    end_time_entry.pack()

    submit_button = tk.Button(end_time_window, text="Submit", command=lambda: self.generate_subset(location_entry.get(), start_time_entry.get(), end_time_entry.get()))
    submit_button.pack()

def generate_subset(self, location, start_time, end_time):
    subset = self.data[(self.data["локация"] == location) & (self.data["время"] >= start_time) & (self.data["время"] <= end_time)]

    subset.plot(x="день", y="температура", kind="line")
    plt.show()


    def visualize_data():
        messagebox.showinfo('Визуализация данных', 'Визуализация данных...')

    def make_predictions():
        messagebox.showinfo('Прогнозирование данных', 'Прогнозирование данных...')

        input_data = torch.tensor([
            [1, 1, 1, 25, 70, 15, 1, 1010, 20],
            [0, 2, 2, 28, 60, 10, 2, 1015, 40],
            [0, 3, 1, 15, 90, 5, 3, 1005, 80],
        ], dtype=torch.float32)

        output_data = torch.tensor([
            [1],
            [0],
            [0],
        ], dtype=torch.float32)

        model = Net()
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(50):
            optimizer.zero_grad()
            outputs = model(input_data)
            loss = criterion(outputs, output_data)
            loss.backward()
            optimizer.step()

        test_input = torch.tensor([
            [1, 1, 3, 20, 75, 10, 3, 1008, 30]
        ], dtype=torch.float32)

        print(model(test_input).detach().numpy())

    button_analyze = tk.Button(window, text='Анализ', command=analyze_data)
    button_analyze.pack()
    
    button_visualize = tk.Button(window, text='Визуализация', command=visualize_data)
    button_visualize.pack()

    button_monitor = tk.Button(window, text='Мониторинг', command=data)
    button_monitor.pack()

    button_predictions = tk.Button(window, text='Прогнозирование', command=make_predictions)
    button_predictions.pack()

    button_exit = tk.Button(window, text='Выход', command=window.destroy)
    button_exit.pack()

    window = window.Tk()
    window.mainloop() 
    
if __name__ == "__main__":
    login_window()
    main_menu()
    