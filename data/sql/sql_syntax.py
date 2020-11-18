from sqlalchemy import create_engine
from sqlalchemy import Column, Integer, String, Float, MetaData, Table, Binary
from sqlalchemy.sql import select, and_
from sqlalchemy.sql.sqltypes import util, processors
import array


class CorrectedBinary(Binary):
    """
    Overrided Binary class from sqlalchemy.
    Ovverided part:
            def result_processor(self, dialect, coltype):
            def process(value):
                if value is not None:
                    value = bytes(value, 'UTF-8') <-- there was string encoding missing!
                return value
            return process
    """

    def bind_processor(self, dialect):
        if dialect.dbapi is None:
            return None

        DBAPIBinary = dialect.dbapi.Binary

        def process(value):
            if value is not None:
                return DBAPIBinary(value)
            else:
                return None

        return process

    # Python 3 has native bytes() type
    # both sqlite3 and pg8000 seem to return it,
    # psycopg2 as of 2.5 returns 'memoryview'
    if util.py2k:
        def result_processor(self, dialect, coltype):
            if util.jython:
                def process(value):
                    if value is not None:
                        if isinstance(value, array.array):
                            return value.tostring()
                        return str(value)
                    else:
                        return None
            else:
                process = processors.to_str
            return process
    else:
        def result_processor(self, dialect, coltype):
            def process(value):
                if value is not None:
                    if 'bytes' in str(type(value)):
                        pass
                    else:
                        value = bytes(value, 'UTF-8')
                return value

            return process


class SqlSyntax():
    """
    :param db_url: [str] url to database (https://docs.sqlalchemy.org/en/13/core/engines.html#database-urls)
    """

    def __init__(self, db_url, verbose=False):
        self.verbose = verbose
        self._db_url = db_url
        self._engine = create_engine(self._db_url, echo=self.verbose)
        
    @staticmethod
    def initiate_market_minutes_table(table_name):
        metadata = MetaData()
        table = Table(table_name, metadata,
                      Column('trade_date', String(30), nullable=False, primary_key=True),
                      Column('Price_low', Float, nullable=False),
                      Column('Price_high', Float, nullable=False),
                      Column('Sell_Volume', Float, nullable=False),
                      Column('Buy_Volume', Float, nullable=False),
                      Column('Price', Float, nullable=False),
                      Column('Open', Float, nullable=False),
                      Column('Closed', Float, nullable=False)
                      )
        return table

    # TODO: Implement high level table creation function

    def create_market_raw_table(self, table_name):
        """
        Create raw market table if not exist.
        :param table_name: [str] Market table name eg. BTC_USDT
        """
        metadata = MetaData()
        table = Table(table_name, metadata,
                      Column('trade_date', String(30), nullable=False),
                      Column('price', Float, nullable=False),
                      Column('quantity', Float, nullable=False),
                      Column('side', String(4), nullable=False),
                      Column('trade_ID', Integer, nullable=False, primary_key=True)
                      )
        metadata.create_all(create_engine(self._db_url, echo=self.verbose))
        return table

    def create_market_minutes_table(self, table_name):
        """
        Create processed minutes market table if not exist.
        :param table_name: [str] Market table name eg. BTC_USDT_minutes
        """
        metadata = MetaData()
        table = Table(table_name, metadata,
                      Column('trade_date', String(30), nullable=False, primary_key=True),
                      Column('Price_low', Float, nullable=False),
                      Column('Price_high', Float, nullable=False),
                      Column('Sell_Volume', Float, nullable=False),
                      Column('Buy_Volume', Float, nullable=False),
                      Column('Price', Float, nullable=False),
                      Column('Open', Float, nullable=False),
                      Column('Closed', Float, nullable=False)
                      )
        metadata.create_all(create_engine(self._db_url, echo=self.verbose))
        return table

    def insert(self, table, values, conflict_method='on_duplicate_ignore'):
        """
        Insert many values into a specific table.
        :param table: [sqlalchemy.Table] Reference for a specific Table
        :param values: [list of dicts] List of dictionaries to insert. (Contains columns names and values)
        :param conflict_method: [str] Method for INSERT when there is a duplicate.
                - on_duplicate_nothing (do nothing, just raise an exception)
                - on_duplicate_ignore (ignore duplicates)
        :return: rowcount [int] Number of rows executed (inserted)
        """
        with self._engine.begin() as conn:
            if conflict_method == 'on_duplicate_nothing':
                result = conn.execute(table.insert(), values)
            elif conflict_method == 'on_duplicate_ignore':
                result = conn.execute(table.insert().prefix_with('OR IGNORE'), values)
        return result.rowcount

    def update(self, table, values, condition):
        """
        Update many values in a specific table.
        :param table: [sqlalchemy.Table] Reference for a specific Table
        :param values: [list of dicts] List of dictionaries to insert.
                (Contains columns names /without conditional column/ and values)
        :param condition [dict] Dictionary of conditional value
                dict(condition_column_name: condition_value)
        :return: rowcount [int] Number of rows executed (updated)
        """
        with self._engine.begin() as conn:
            result = conn.execute(table.update().where(table.c[[*condition.keys()][0]] == [*condition.values()][0]),
                                  values)
        return result.rowcount


    def select_min_last_data(self, table, number_of_rows):
        """
        Selects non duplicated last /n/ rows from minutes database with ascending order
        :param table: [sqlalchemy.Table] Reference for a specific Table
        :param number_of_rows: [int] Number of rows to SELECT from Database
        :return: [list of rows[sets]] List containing each SELECTed row as a set()
        """
        sub_query = select([table.c.trade_date.distinct().label('trade_date'),
                            table.c.Price_low.label('Price_low'),
                            table.c.Price_high.label('Price_high'),
                            table.c.Sell_Volume.label('Sell_Volume'),
                            table.c.Buy_Volume.label('Buy_Volume'),
                            table.c.Price.label('Price'),
                            table.c.Open.label('Open'),
                            table.c.Closed.label('Closed')]).order_by(table.c.trade_date.desc()).limit(number_of_rows)
        query = select([sub_query.c.trade_date.distinct().label('trade_date'),
                        sub_query.c.Price_low.label('Price_low'),
                        sub_query.c.Price_high.label('Price_high'),
                        sub_query.c.Sell_Volume.label('Sell_Volume'),
                        sub_query.c.Buy_Volume.label('Buy_Volume'),
                        sub_query.c.Price.label('Price'),
                        sub_query.c.Open.label('Open'),
                        sub_query.c.Closed.label('Closed')]).order_by(sub_query.c.trade_date.asc())
        with self._engine.begin() as conn:
            result = conn.execute(query)
            results = [row for row in result]
        return results

    def select_min_price_by_time(self, table, from_time=None, to_time=None, timestamp=False):
        """
        Selects all prices within given time range from market minute table
        :param table: [sqlalchemy.Table] Reference for a specific Table
        :param from_time: start datetime
        :param to_time: end datetime
        :return: [list of rows[sets]] List containing each SELECTed row as a set()
        """
        if from_time is None and to_time is None:
            query = select([table.c.Price.label('Price')]).order_by(table.c.trade_date.asc())
        elif from_time is None:
            query = select([table.c.Price.label('Price')]).order_by(table.c.trade_date.asc()).where(
                table.c.trade_date <= to_time)
        elif to_time is None:
            query = select([table.c.Price.label('Price')]).order_by(table.c.trade_date.asc()).where(
                table.c.trade_date >= from_time)
        else:
            if not timestamp:
                query = select([table.c.Price.label('Price')]).order_by(table.c.trade_date.asc()).where(
                    and_(table.c.trade_date >= from_time, table.c.trade_date <= to_time))
            else:
                query = select([table.c.trade_date,
                                table.c.Price.label('Price')]).order_by(table.c.trade_date.asc()).where(
                    and_(table.c.trade_date >= from_time, table.c.trade_date <= to_time))
        with self._engine.begin() as conn:
            result = conn.execute(query)
            results = [row for row in result]
        return results

    def select_min_by_time(self, table, from_time=None, to_time=None):
        """
        Selects all prices, volumes etc. within given time range from market minute table
        :param table: [sqlalchemy.Table] Reference for a specific Table
        :param from_time: start datetime
        :param to_time: end datetime
        :return: [list of rows[sets]] List containing each SELECTed row as a set()
        """
        if from_time is None and to_time is None:
            query = select([table.c.trade_date.label('trade_date'),
                            table.c.Price_low.label('Price_low'),
                            table.c.Price_high.label('Price_high'),
                            table.c.Sell_Volume.label('Sell_Volume'),
                            table.c.Buy_Volume.label('Buy_Volume'),
                            table.c.Price.label('Price'),
                            table.c.Open.label('Open'),
                            table.c.Closed.label('Closed')]).order_by(table.c.trade_date.asc())
        elif from_time is None:
            query = select([table.c.trade_date.label('trade_date'),
                            table.c.Price_low.label('Price_low'),
                            table.c.Price_high.label('Price_high'),
                            table.c.Sell_Volume.label('Sell_Volume'),
                            table.c.Buy_Volume.label('Buy_Volume'),
                            table.c.Price.label('Price'),
                            table.c.Open.label('Open'),
                            table.c.Closed.label('Closed')]).order_by(table.c.trade_date.asc()).where(
                table.c.trade_date <= to_time)
        elif to_time is None:
            query = select([table.c.trade_date.label('trade_date'),
                            table.c.Price_low.label('Price_low'),
                            table.c.Price_high.label('Price_high'),
                            table.c.Sell_Volume.label('Sell_Volume'),
                            table.c.Buy_Volume.label('Buy_Volume'),
                            table.c.Price.label('Price'),
                            table.c.Open.label('Open'),
                            table.c.Closed.label('Closed')]).order_by(table.c.trade_date.asc()).where(
                table.c.trade_date >= from_time)
        else:
            query = select([table.c.trade_date.label('trade_date'),
                            table.c.Price_low.label('Price_low'),
                            table.c.Price_high.label('Price_high'),
                            table.c.Sell_Volume.label('Sell_Volume'),
                            table.c.Buy_Volume.label('Buy_Volume'),
                            table.c.Price.label('Price'),
                            table.c.Open.label('Open'),
                            table.c.Closed.label('Closed')]).order_by(table.c.trade_date.asc()).where(
                and_(table.c.trade_date >= from_time, table.c.trade_date <= to_time))
        with self._engine.begin() as conn:
            result = conn.execute(query)
            results = [row for row in result]
        return results

