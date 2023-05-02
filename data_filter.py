import os

category = 'business'
dir_path = './data/bbc_news_data/article/' + category + '/'
sum_path = './data/bbc_news_data/summaries/' + category + '/'

files_to_del = []

for filename in os.listdir(dir_path):
    
    if filename.endswith('.txt'):
        file_path = os.path.join(dir_path, filename)
        summary_path = os.path.join(sum_path, filename)
        with open(file_path, 'r') as f:
            words = f.read().split()
            
            if len(words) < 300:

                print(str(file_path) + " is less than 300 words.")
                files_to_del.append(file_path)
                files_to_del.append(summary_path)
            f.close()



for file_name in files_to_del:
    if(os.path.exists(file_name)):
        print(f"{file_name} has been deleted.")
        os.remove(file_name)