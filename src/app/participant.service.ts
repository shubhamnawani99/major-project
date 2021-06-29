import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { Participant } from './participant';

@Injectable({
  providedIn: 'root'
})
export class ParticipantService {
  baseUrl: string = 'http://localhost:5000'

  constructor(private http: HttpClient) { }

  getAllParticipants():Observable<Participant[]>{
    return this.http.get<Participant[]>(`${this.baseUrl}/participants`)
  }

  start(): Observable<any>{
    return this.http.get(`${this.baseUrl}/start`)
  }
}
