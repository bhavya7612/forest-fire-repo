import mysql.connector as connection

class mysqlconnector:
    def __init__(self):
        try:
            self.conn=connection.connect(host="localhost", user="root", database="forestfiredb", password="root")
            self.cur=self.conn.cursor()
        except:
            print("Could not connect to SQL")
    
    def user_signup(self,username,email,password):
        self.cur.execute(f"insert into users(username,email,password) values('{username}','{email}','{password}');")
        self.conn.commit()
        self.cur.execute(f"select * from users where username='{username}' and password='{password}';")
        res=self.cur.fetchall()
        print(res)
        return res
    
    def user_exists_signup(self,email):
        self.cur.execute(f"select * from users where email='{email}';")
        res=self.cur.fetchall()
        print(res)
        return res
    
    def user_login(self,email,password):
        self.cur.execute(f"select * from users where email='{email}' and password='{password}';")
        res=self.cur.fetchall()
        print(res)
        return res