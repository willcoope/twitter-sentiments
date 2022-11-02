import matplotlib.pyplot as plt
import csv
  
x1 = []
y1 = []
  
with open('TSLA_monthly.csv') as file_obj:
    next(file_obj)
    # Create reader object by passing the file object to reader method
    reader_obj = csv.reader(file_obj)
    for row in reader_obj:
        #print((row[0]))
        x1.append(row[0])
        y1.append(float(row[4]))
  
plt.plot(x1, y1, color = 'g', label = "Price")
plt.bar(x1, y1, color = 'r', label = "Sentiment")
ticks = []
plt.xticks(rotation=45)
plt.tight_layout()
plt.xlabel('Date')
plt.tick_params(labelsize=4)
plt.ylabel('Price')
plt.title('Stock Price')
plt.legend()
plt.show()