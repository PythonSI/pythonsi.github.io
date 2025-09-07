{{ fullname | escape | underline}}

.. automodule:: {{ fullname }}
   :members:
   
   {% block functions %}
   {% if functions %}
   
   Functions
   ---------
   
   {% for item in functions %}
   .. autofunction:: {{ item }}
      
   .. raw:: html

	       <div class="sphx-glr-clear"></div>
   {%- endfor %}
   {% endif %}
   {% endblock %}
   
   {% block classes %}
   {% if classes %}
   
   
   
   Classes
   -------
   
   {% for item in classes %}
   .. autoclass:: {{ item }}
      :members:
      :special-members: __call__
   .. raw:: html

	       <div class="sphx-glr-clear"></div>
   {%- endfor %}
   {% endif %}
   {% endblock %}
   
   {% block exceptions %}
   {% if exceptions %}
   
   
   
   Exceptions
   ----------
   
   {% for item in exceptions %}
   .. autoexception:: {{ item }}
      :members:
   .. raw:: html

	       <div class="sphx-glr-clear"></div>
   {%- endfor %}
   {% endif %}
   {% endblock %}